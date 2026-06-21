use super::dispatch;
use super::provider_init::ProviderChoice;
use crate::protocol::{Request, ServerEvent};
use crate::provider::Provider;
use crate::transport::{ReadHalf, WriteHalf};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

const ACP_PROTOCOL_VERSION: u64 = 1;

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_INVALID_REQUEST: i64 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INVALID_PARAMS: i64 = -32602;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;
const JSONRPC_SERVER_ERROR: i64 = -32000;

const AUTOCOMPLETE_SYSTEM_PROMPT: &str = "You are a low-latency inline code completion engine. Return only the exact text to insert at the cursor. Do not repeat the existing prefix or suffix. Do not explain. Do not use markdown fences.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AcpProfile {
    Standard,
    Extended,
    Full,
}

impl AcpProfile {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "extended" => Self::Extended,
            "full" => Self::Full,
            _ => Self::Standard,
        }
    }

    fn is_extended(self) -> bool {
        matches!(self, Self::Extended | Self::Full)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Extended => "extended",
            Self::Full => "full",
        }
    }
}

#[derive(Debug)]
struct JsonRpcMessage {
    id: Option<Value>,
    method: Option<String>,
    params: Value,
}

impl JsonRpcMessage {
    fn parse(line: &str) -> std::result::Result<Self, (i64, String)> {
        let value: Value =
            serde_json::from_str(line).map_err(|err| (JSONRPC_PARSE_ERROR, err.to_string()))?;
        let object = value.as_object().ok_or_else(|| {
            (
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must be an object".to_string(),
            )
        })?;
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err((
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must include jsonrpc=\"2.0\"".to_string(),
            ));
        }
        Ok(Self {
            id: object.get("id").cloned(),
            method: object
                .get("method")
                .and_then(Value::as_str)
                .map(str::to_string),
            params: object.get("params").cloned().unwrap_or(Value::Null),
        })
    }
}

struct DaemonSession {
    session_id: String,
    reader: Mutex<BufReader<ReadHalf>>,
    writer: Mutex<WriteHalf>,
    next_request_id: AtomicU64,
    active_prompt_id: Mutex<Option<u64>>,
    prompt_running: AtomicBool,
}

impl DaemonSession {
    fn new(session_id: String, reader: ReadHalf, writer: WriteHalf, next_request_id: u64) -> Self {
        Self {
            session_id,
            reader: Mutex::new(BufReader::new(reader)),
            writer: Mutex::new(writer),
            next_request_id: AtomicU64::new(next_request_id),
            active_prompt_id: Mutex::new(None),
            prompt_running: AtomicBool::new(false),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn send(&self, request: &Request) -> Result<()> {
        let mut json = serde_json::to_string(request)?;
        json.push('\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.flush().await?;
        Ok(())
    }

    async fn read_event(&self) -> Result<ServerEvent> {
        let mut line = String::new();
        let mut reader = self.reader.lock().await;
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("Jcode daemon disconnected");
        }
        let event = serde_json::from_str(&line)
            .with_context(|| format!("failed to decode Jcode daemon event: {}", line.trim_end()))?;
        Ok(event)
    }
}

#[derive(Clone)]
struct AcpRuntime {
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    sessions: Arc<Mutex<HashMap<String, Arc<DaemonSession>>>>,
    autocomplete_tasks: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    profile: AcpProfile,
    provider_choice: ProviderChoice,
    model: Option<String>,
    provider_profile: Option<String>,
}

impl AcpRuntime {
    fn new(
        profile: AcpProfile,
        provider_choice: ProviderChoice,
        model: Option<String>,
        provider_profile: Option<String>,
    ) -> Self {
        Self {
            stdout: Arc::new(Mutex::new(tokio::io::stdout())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            autocomplete_tasks: Arc::new(Mutex::new(HashMap::new())),
            profile,
            provider_choice,
            model,
            provider_profile,
        }
    }

    async fn run(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                return Ok(());
            }
            if line.trim().is_empty() {
                continue;
            }

            let message = match JsonRpcMessage::parse(&line) {
                Ok(message) => message,
                Err((code, message)) => {
                    self.write_error_value(
                        Value::Null,
                        code,
                        format!("Invalid JSON-RPC request: {message}"),
                    )
                    .await?;
                    continue;
                }
            };

            self.handle_message(message).await?;
        }
    }

    async fn handle_message(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(method) = message.method.as_deref() else {
            if let Some(id) = message.id {
                self.write_error_value(
                    id,
                    JSONRPC_INVALID_REQUEST,
                    "JSON-RPC request missing method".to_string(),
                )
                .await?;
            }
            return Ok(());
        };

        match method {
            "initialize" => {
                if let Some(id) = message.id {
                    self.write_result(id, initialize_result(&message.params, self.profile))
                        .await?;
                }
            }
            "session/new" => self.handle_session_new(message).await?,
            "session/load" => self.handle_session_load(message, true).await?,
            "session/resume" => self.handle_session_load(message, false).await?,
            "session/prompt" => self.handle_session_prompt(message).await?,
            "session/cancel" => self.handle_session_cancel(message).await?,
            "session/close" => self.handle_session_close(message).await?,
            "session/capabilities" => self.handle_session_capabilities(message).await?,
            "workspace/autocomplete" => self.handle_workspace_autocomplete(message).await?,
            "workspace/autocomplete/cancel" => {
                self.handle_workspace_autocomplete_cancel(message).await?
            }
            _ if method.starts_with('_') => {
                if let Some(id) = message.id {
                    self.write_error_value(
                        id,
                        JSONRPC_METHOD_NOT_FOUND,
                        format!("Unsupported Jcode ACP extension method: {method}"),
                    )
                    .await?;
                }
            }
            _ => {
                if let Some(id) = message.id {
                    self.write_error_value(
                        id,
                        JSONRPC_METHOD_NOT_FOUND,
                        format!("Unsupported ACP method: {method}"),
                    )
                    .await?;
                }
            }
        }

        Ok(())
    }

    async fn handle_session_new(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let cwd = match cwd_from_params(&message.params) {
            Ok(cwd) => cwd,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Err(err) = ensure_no_acp_mcp_servers(&message.params) {
            self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                .await?;
            return Ok(());
        }

        match self.create_new_session(cwd).await {
            Ok(session) => {
                let session_id = session.session_id.clone();
                self.sessions
                    .lock()
                    .await
                    .insert(session_id.clone(), Arc::new(session));
                self.write_result(id, json!({ "sessionId": session_id }))
                    .await?;
            }
            Err(err) => {
                self.write_error_value(
                    id,
                    JSONRPC_INTERNAL_ERROR,
                    format!("Failed to create Jcode session: {err:#}"),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn handle_session_load(
        &self,
        message: JsonRpcMessage,
        replay_history: bool,
    ) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let cwd = match cwd_from_params(&message.params) {
            Ok(cwd) => cwd,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Err(err) = ensure_no_acp_mcp_servers(&message.params) {
            self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                .await?;
            return Ok(());
        }

        match self
            .attach_existing_session(session_id.clone(), cwd, replay_history)
            .await
        {
            Ok(session) => {
                self.sessions
                    .lock()
                    .await
                    .insert(session.session_id.clone(), Arc::new(session));
                self.write_result(id, json!({})).await?;
            }
            Err(err) => {
                self.write_error_value(
                    id,
                    JSONRPC_INTERNAL_ERROR,
                    format!("Failed to attach Jcode session '{session_id}': {err:#}"),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn handle_session_prompt(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let (text, images) = match prompt_from_params(&message.params) {
            Ok(prompt) => prompt,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let session = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).cloned()
        };
        let Some(session) = session else {
            self.write_error_value(
                id,
                JSONRPC_INVALID_PARAMS,
                format!("Unknown ACP session id: {session_id}"),
            )
            .await?;
            return Ok(());
        };

        if session
            .prompt_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            self.write_error_value(
                id,
                JSONRPC_SERVER_ERROR,
                format!("Session {session_id} is already processing a prompt"),
            )
            .await?;
            return Ok(());
        }

        let runtime = self.clone();
        tokio::spawn(async move {
            let result = runtime.run_prompt(id.clone(), session, text, images).await;
            if let Err(err) = result {
                let _ = runtime
                    .write_error_value(
                        id,
                        JSONRPC_INTERNAL_ERROR,
                        format!("Prompt failed: {err:#}"),
                    )
                    .await;
            }
        });
        Ok(())
    }

    async fn handle_session_cancel(&self, message: JsonRpcMessage) -> Result<()> {
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                if let Some(id) = message.id {
                    self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                        .await?;
                }
                return Ok(());
            }
        };
        let session = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).cloned()
        };
        if let Some(session) = session {
            let cancel_id = session.next_id();
            let _ = session.send(&Request::Cancel { id: cancel_id }).await;
        }
        if let Some(id) = message.id {
            self.write_result(id, json!({})).await?;
        }
        Ok(())
    }

    async fn handle_session_close(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Some(session) = self.sessions.lock().await.remove(&session_id) {
            let cancel_id = session.next_id();
            let _ = session.send(&Request::Cancel { id: cancel_id }).await;
        }
        self.write_result(id, json!({})).await?;
        Ok(())
    }

    async fn handle_session_capabilities(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err).await?;
                return Ok(());
            }
        };

        if !self.sessions.lock().await.contains_key(&session_id) {
            self.write_error_value(
                id,
                JSONRPC_INVALID_PARAMS,
                format!("Unknown ACP sessionId: {session_id}"),
            )
            .await?;
            return Ok(());
        }

        let result = self.build_effective_capabilities_snapshot(&session_id).await;
        self.write_result(id, result).await?;
        Ok(())
    }

    async fn handle_workspace_autocomplete(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let request = match autocomplete_request_from_params(&message.params) {
            Ok(request) => request,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };

        let runtime = self.clone();
        let rpc_id = id.clone();
        let task_key = rpc_tracking_key(&id);
        let task_key_for_cleanup = task_key.clone();
        let handle = tokio::spawn(async move {
            let result = runtime.execute_autocomplete_rpc(request).await;
            match result {
                Ok(response) => {
                    let _ = runtime.write_result(rpc_id.clone(), response).await;
                }
                Err(err) => {
                    let _ = runtime.write_common_error(rpc_id.clone(), err).await;
                }
            }
            runtime
                .autocomplete_tasks
                .lock()
                .await
                .remove(&task_key_for_cleanup);
        });

        self.autocomplete_tasks
            .lock()
            .await
            .insert(task_key, handle);
        Ok(())
    }

    async fn handle_workspace_autocomplete_cancel(&self, message: JsonRpcMessage) -> Result<()> {
        let request_id = match autocomplete_cancel_request_id(&message.params) {
            Ok(request_id) => request_id,
            Err(err) => {
                if let Some(id) = message.id {
                    self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                        .await?;
                }
                return Ok(());
            }
        };

        if let Some(task) = self.autocomplete_tasks.lock().await.remove(&request_id) {
            task.abort();
        }

        if let Some(id) = message.id {
            self.write_result(id, json!({})).await?;
        }
        Ok(())
    }

    async fn ensure_daemon(&self) -> Result<()> {
        if dispatch::server_is_running().await {
            return Ok(());
        }
        dispatch::spawn_server(
            &self.provider_choice,
            self.model.as_deref(),
            self.provider_profile.as_deref(),
        )
        .await
    }

    async fn connect_daemon(&self) -> Result<(ReadHalf, WriteHalf)> {
        self.ensure_daemon().await?;
        let stream = crate::server::connect_socket(&crate::server::socket_path()).await?;
        Ok(stream.into_split())
    }

    async fn create_new_session(&self, cwd: PathBuf) -> Result<DaemonSession> {
        let (reader, writer) = self.connect_daemon().await?;
        let session = DaemonSession::new(String::new(), reader, writer, 2);
        let subscribe_id = 1;
        session
            .send(&Request::Subscribe {
                id: subscribe_id,
                working_dir: Some(cwd.display().to_string()),
                selfdev: None,
                target_session_id: None,
                client_instance_id: Some("acp".to_string()),
                client_has_local_history: false,
                allow_session_takeover: false,
            })
            .await?;
        wait_for_done(&session, subscribe_id).await?;
        let history = request_history(&session).await?;
        let session_id = match history {
            ServerEvent::History { session_id, .. } => session_id,
            other => anyhow::bail!("expected history after session creation, got {other:?}"),
        };
        Ok(DaemonSession::new(
            session_id,
            session.reader.into_inner().into_inner(),
            session.writer.into_inner(),
            session.next_request_id.load(Ordering::Relaxed),
        ))
    }

    async fn attach_existing_session(
        &self,
        target_session_id: String,
        _cwd: PathBuf,
        replay_history: bool,
    ) -> Result<DaemonSession> {
        let (reader, writer) = self.connect_daemon().await?;
        let session = DaemonSession::new(String::new(), reader, writer, 2);
        let resume_id = 1;
        session
            .send(&Request::ResumeSession {
                id: resume_id,
                session_id: target_session_id.clone(),
                client_instance_id: Some("acp".to_string()),
                client_has_local_history: false,
                allow_session_takeover: false,
            })
            .await?;

        let mut attached_id = target_session_id;
        loop {
            let event = session.read_event().await?;
            match event {
                ServerEvent::Ack { .. } => {}
                ServerEvent::History {
                    session_id,
                    messages,
                    ..
                } => {
                    attached_id = session_id.clone();
                    if replay_history {
                        self.replay_history(&session_id, messages).await?;
                    }
                }
                ServerEvent::Done { id } if id == resume_id => break,
                ServerEvent::Error { id, message, .. } if id == resume_id => {
                    anyhow::bail!(message);
                }
                other => {
                    if self.profile.is_extended() {
                        self.write_jcode_extension_event(&attached_id, &other)
                            .await?;
                    }
                }
            }
        }

        Ok(DaemonSession::new(
            attached_id,
            session.reader.into_inner().into_inner(),
            session.writer.into_inner(),
            session.next_request_id.load(Ordering::Relaxed),
        ))
    }

    async fn replay_history(
        &self,
        session_id: &str,
        messages: Vec<crate::protocol::HistoryMessage>,
    ) -> Result<()> {
        for message in messages {
            let update_name = match message.role.as_str() {
                "user" => "user_message_chunk",
                "assistant" => "agent_message_chunk",
                _ => "agent_message_chunk",
            };
            self.write_notification(
                "session/update",
                json!({
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": update_name,
                        "content": {
                            "type": "text",
                            "text": message.content,
                        }
                    }
                }),
            )
            .await?;
        }
        Ok(())
    }

    async fn run_prompt(
        &self,
        rpc_id: Value,
        session: Arc<DaemonSession>,
        text: String,
        images: Vec<(String, String)>,
    ) -> Result<()> {
        let prompt_id = session.next_id();
        {
            let mut active = session.active_prompt_id.lock().await;
            *active = Some(prompt_id);
        }

        let send_result = session
            .send(&Request::Message {
                id: prompt_id,
                content: text,
                images,
                system_reminder: None,
                disable_tools: None,
            })
            .await;
        if let Err(err) = send_result {
            cleanup_prompt_state(&session).await;
            return Err(err);
        }

        let mut mapper = EventMapper::new(session.session_id.clone(), self.profile);
        let mut stop_reason = "end_turn".to_string();
        loop {
            let event = match session.read_event().await {
                Ok(event) => event,
                Err(err) => {
                    cleanup_prompt_state(&session).await;
                    return Err(err);
                }
            };
            if self.profile.is_extended() {
                self.write_jcode_extension_event(&session.session_id, &event)
                    .await?;
            }
            match event {
                ServerEvent::Ack { .. } => {}
                ServerEvent::Done { id } if id == prompt_id => break,
                ServerEvent::Interrupted => {
                    stop_reason = "cancelled".to_string();
                }
                ServerEvent::Error { id, message, .. } if id == prompt_id => {
                    cleanup_prompt_state(&session).await;
                    self.write_error_value(rpc_id, JSONRPC_SERVER_ERROR, message)
                        .await?;
                    return Ok(());
                }
                other => {
                    for update in mapper.map_event(other) {
                        self.write_notification(
                            "session/update",
                            json!({
                                "sessionId": session.session_id,
                                "update": update,
                            }),
                        )
                        .await?;
                    }
                }
            }
        }

        cleanup_prompt_state(&session).await;
        self.write_result(rpc_id, json!({ "stopReason": stop_reason }))
            .await?;
        Ok(())
    }

    async fn write_result(&self, id: Value, result: Value) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))
        .await
    }

    async fn write_error_value(&self, id: Value, code: i64, message: String) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message,
            }
        }))
        .await
    }

    async fn write_common_error(&self, id: Value, error: AutocompleteRpcError) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": error.jsonrpc_code,
                "message": error.message,
                "data": {
                    "code": error.common_code,
                    "retryable": error.retryable,
                }
            }
        }))
        .await
    }

    async fn write_notification(&self, method: &str, params: Value) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn write_jcode_extension_event(
        &self,
        session_id: &str,
        event: &ServerEvent,
    ) -> Result<()> {
        self.write_notification(
            "_jcode/server_event",
            json!({
                "sessionId": session_id,
                "event": serde_json::to_value(event).unwrap_or(Value::Null),
            }),
        )
        .await
    }

    async fn write_value(&self, value: Value) -> Result<()> {
        let mut stdout = self.stdout.lock().await;
        let mut line = serde_json::to_string(&value)?;
        line.push('\n');
        stdout.write_all(line.as_bytes()).await?;
        stdout.flush().await?;
        Ok(())
    }

    async fn execute_autocomplete_rpc(
        &self,
        request: AutocompleteRequest,
    ) -> std::result::Result<Value, AutocompleteRpcError> {
        let provider =
            super::provider_init::init_provider_quiet(&self.provider_choice, self.model.as_deref())
                .await
                .map_err(|err| {
                    AutocompleteRpcError::server(
                        "AUTOCOMPLETE_UNAVAILABLE",
                        format!("Autocomplete provider is unavailable: {err}"),
                        false,
                    )
                })?;

        run_autocomplete_request(provider, request).await
    }

    async fn build_effective_capabilities_snapshot(&self, session_id: &str) -> Value {
        let provider_result =
            super::provider_init::init_provider_quiet(&self.provider_choice, self.model.as_deref())
                .await
                .map_err(|err| err.to_string());

        build_effective_capabilities_snapshot(
            self.profile,
            &self.provider_choice,
            self.model.as_deref(),
            self.provider_profile.as_deref(),
            session_id,
            provider_result,
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
struct AutocompleteRequest {
    #[serde(rename = "sessionId")]
    session_id: String,
    document: AutocompleteDocument,
    cursor: AutocompleteCursor,
    limits: AutocompleteLimits,
}

#[derive(Debug, Clone, Deserialize)]
struct AutocompleteDocument {
    uri: String,
    #[serde(rename = "languageId")]
    language_id: String,
    version: u64,
    prefix: String,
    suffix: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AutocompleteLimits {
    #[serde(rename = "maxPrefixChars")]
    max_prefix_chars: usize,
    #[serde(rename = "maxSuffixChars")]
    max_suffix_chars: usize,
    #[serde(rename = "timeoutMs")]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct AutocompleteCursor {
    line: u64,
    character: u64,
}

impl AutocompleteDocument {
    fn cursor_line(&self, cursor: &AutocompleteCursor) -> u64 {
        cursor.line
    }

    fn cursor_character(&self, cursor: &AutocompleteCursor) -> u64 {
        cursor.character
    }
}

#[derive(Debug, Clone)]
struct AutocompleteRpcError {
    jsonrpc_code: i64,
    common_code: &'static str,
    message: String,
    retryable: bool,
}

impl AutocompleteRpcError {
    fn server(common_code: &'static str, message: String, retryable: bool) -> Self {
        Self {
            jsonrpc_code: JSONRPC_SERVER_ERROR,
            common_code,
            message,
            retryable,
        }
    }
}

async fn cleanup_prompt_state(session: &DaemonSession) {
    {
        let mut active = session.active_prompt_id.lock().await;
        *active = None;
    }
    session.prompt_running.store(false, Ordering::SeqCst);
}

async fn wait_for_done(session: &DaemonSession, request_id: u64) -> Result<()> {
    loop {
        match session.read_event().await? {
            ServerEvent::Ack { .. } => {}
            ServerEvent::Done { id } if id == request_id => return Ok(()),
            ServerEvent::Error { id, message, .. } if id == request_id => anyhow::bail!(message),
            _ => {}
        }
    }
}

async fn request_history(session: &DaemonSession) -> Result<ServerEvent> {
    let id = session.next_id();
    session.send(&Request::GetHistory { id }).await?;
    loop {
        match session.read_event().await? {
            ServerEvent::Ack { .. } => {}
            event @ ServerEvent::History { id: event_id, .. } if event_id == id => {
                return Ok(event);
            }
            ServerEvent::Error {
                id: event_id,
                message,
                ..
            } if event_id == id => anyhow::bail!(message),
            _ => {}
        }
    }
}

struct EventMapper {
    session_id: String,
    profile: AcpProfile,
    current_tool_id: Option<String>,
    tool_inputs: HashMap<String, String>,
}

impl EventMapper {
    fn new(session_id: String, profile: AcpProfile) -> Self {
        Self {
            session_id,
            profile,
            current_tool_id: None,
            tool_inputs: HashMap::new(),
        }
    }

    fn map_event(&mut self, event: ServerEvent) -> Vec<Value> {
        match event {
            ServerEvent::TextDelta { text } => vec![agent_message_chunk(text)],
            ServerEvent::TextReplace { text } => vec![agent_message_chunk(text)],
            ServerEvent::ToolStart { id, name } => {
                self.current_tool_id = Some(id.clone());
                self.tool_inputs.entry(id.clone()).or_default();
                vec![json!({
                    "sessionUpdate": "tool_call",
                    "toolCallId": id,
                    "title": tool_title(&name),
                    "kind": tool_kind(&name),
                    "status": "pending",
                })]
            }
            ServerEvent::ToolInput { delta } => {
                let Some(tool_id) = self.current_tool_id.clone() else {
                    return Vec::new();
                };
                let buffer = self.tool_inputs.entry(tool_id.clone()).or_default();
                buffer.push_str(&delta);
                let mut update = json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_id,
                });
                if let Some(raw_input) = parse_json_object(buffer)
                    && let Some(object) = update.as_object_mut()
                {
                    object.insert("rawInput".to_string(), raw_input);
                }
                vec![update]
            }
            ServerEvent::ToolExec { id, name } => {
                self.current_tool_id = Some(id.clone());
                let mut update = json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": id,
                    "title": tool_title(&name),
                    "kind": tool_kind(&name),
                    "status": "in_progress",
                });
                if let Some(input) = self
                    .tool_inputs
                    .get(update["toolCallId"].as_str().unwrap_or_default())
                    && let Some(raw_input) = parse_json_object(input)
                    && let Some(object) = update.as_object_mut()
                {
                    object.insert("rawInput".to_string(), raw_input);
                }
                vec![update]
            }
            ServerEvent::ToolDone {
                id,
                name,
                output,
                error,
            } => vec![json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": id,
                "title": tool_title(&name),
                "kind": tool_kind(&name),
                "status": if error.is_some() { "failed" } else { "completed" },
                "content": [{
                    "type": "content",
                    "content": {
                        "type": "text",
                        "text": output,
                    }
                }],
                "rawOutput": {
                    "output": output,
                    "error": error,
                }
            })],
            ServerEvent::GeneratedImage {
                id,
                path,
                output_format,
                revised_prompt,
                ..
            } => vec![json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": id,
                "status": "completed",
                "content": [{
                    "type": "content",
                    "content": {
                        "type": "text",
                        "text": format!("Generated image: {path} ({output_format}){}", revised_prompt.map(|prompt| format!("\nRevised prompt: {prompt}")).unwrap_or_default()),
                    }
                }]
            })],
            ServerEvent::Compaction { trigger, .. } if self.profile.is_extended() => vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": format!("\n[Jcode compacted context: {trigger}]\n"),
                }
            })],
            ServerEvent::SessionRenamed { display_title, .. } => vec![json!({
                "sessionUpdate": "session_info_update",
                "title": display_title,
            })],
            ServerEvent::McpStatus { servers } if self.profile.is_extended() => vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": format!("\n[Jcode MCP status: {}]\n", servers.join(", ")),
                }
            })],
            _ => {
                let _ = &self.session_id;
                Vec::new()
            }
        }
    }
}

fn parse_json_object(input: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(input).ok()?;
    value.as_object()?;
    Some(value)
}

fn autocomplete_request_from_params(
    params: &Value,
) -> std::result::Result<AutocompleteRequest, String> {
    let request: AutocompleteRequest =
        serde_json::from_value(params.clone()).map_err(|err| err.to_string())?;
    if request.session_id.trim().is_empty() {
        return Err("Missing required sessionId".to_string());
    }
    if request.document.uri.trim().is_empty() {
        return Err("Autocomplete document uri is required".to_string());
    }
    if request.document.language_id.trim().is_empty() {
        return Err("Autocomplete document languageId is required".to_string());
    }
    if request.limits.max_prefix_chars == 0 {
        return Err("limits.maxPrefixChars must be greater than zero".to_string());
    }
    if request.limits.timeout_ms == 0 {
        return Err("limits.timeoutMs must be greater than zero".to_string());
    }
    if request.document.prefix.chars().count() > request.limits.max_prefix_chars
        || request.document.suffix.chars().count() > request.limits.max_suffix_chars
    {
        return Err(format!(
            "Autocomplete input exceeds limits for {}",
            request.document.uri
        ));
    }
    Ok(request)
}

fn autocomplete_cancel_request_id(params: &Value) -> std::result::Result<String, String> {
    params
        .get("requestId")
        .or_else(|| params.get("id"))
        .map(rpc_tracking_key)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing required requestId".to_string())
}

fn rpc_tracking_key(id: &Value) -> String {
    match id {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

async fn run_autocomplete_request(
    provider: Arc<dyn Provider>,
    request: AutocompleteRequest,
) -> std::result::Result<Value, AutocompleteRpcError> {
    let prompt = build_autocomplete_prompt(&request);
    let timeout = std::time::Duration::from_millis(request.limits.timeout_ms);
    let raw_completion = tokio::time::timeout(
        timeout,
        provider.complete_simple(&prompt, AUTOCOMPLETE_SYSTEM_PROMPT),
    )
    .await
    .map_err(|_| {
        AutocompleteRpcError::server(
            "AUTOCOMPLETE_TIMEOUT",
            format!(
                "Autocomplete timed out after {} ms for {}",
                request.limits.timeout_ms, request.document.uri
            ),
            true,
        )
    })?
    .map_err(|err| {
        AutocompleteRpcError::server(
            "AUTOCOMPLETE_UNAVAILABLE",
            format!("Autocomplete request failed: {err}"),
            true,
        )
    })?;

    let completion = normalize_autocomplete_completion(&raw_completion, &request.document.suffix);
    let finish_reason = if completion.is_empty() {
        "empty"
    } else {
        "completed"
    };

    Ok(json!({
        "completion": completion,
        "range": {
            "startLine": request.document.cursor_line(&request.cursor),
            "startCharacter": request.document.cursor_character(&request.cursor),
            "endLine": request.document.cursor_line(&request.cursor),
            "endCharacter": request.document.cursor_character(&request.cursor),
        },
        "confidence": if completion.is_empty() { 0.0 } else { 0.5 },
        "finishReason": finish_reason,
        "providerEffective": {
            "providerName": provider.name(),
            "modelName": provider.model(),
        }
    }))
}

fn build_autocomplete_prompt(request: &AutocompleteRequest) -> String {
    format!(
        "Complete the code at the cursor.\nLanguage: {}\nFile: {}\nDocument version: {}\n\nPrefix:\n<PRE>\n{}\n</PRE>\n\nSuffix:\n<SUF>\n{}\n</SUF>\n\nReturn only the missing text to insert between <PRE> and <SUF>.",
        request.document.language_id,
        request.document.uri,
        request.document.version,
        request.document.prefix,
        request.document.suffix
    )
}

fn normalize_autocomplete_completion(raw: &str, suffix: &str) -> String {
    let mut completion = raw.replace("\r\n", "\n");
    if completion.starts_with("```") {
        completion = strip_markdown_fence(&completion);
    }

    let suffix_chars: Vec<char> = suffix.chars().collect();
    let completion_chars: Vec<char> = completion.chars().collect();
    let max_overlap = completion_chars.len().min(suffix_chars.len());
    let mut overlap = 0usize;
    for candidate in (1..=max_overlap).rev() {
        if completion_chars[completion_chars.len() - candidate..] == suffix_chars[..candidate] {
            overlap = candidate;
            break;
        }
    }

    if overlap > 0 {
        completion_chars[..completion_chars.len() - overlap]
            .iter()
            .collect()
    } else {
        completion
    }
}

fn strip_markdown_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    let Some(without_open) = trimmed.strip_prefix("```") else {
        return raw.to_string();
    };
    let body = without_open
        .split_once('\n')
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    body.strip_suffix("```")
        .unwrap_or(body)
        .trim_end_matches('\n')
        .to_string()
}

fn initialize_result(params: &Value, profile: AcpProfile) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_u64)
        .unwrap_or(ACP_PROTOCOL_VERSION);
    let protocol_version = requested
        .min(ACP_PROTOCOL_VERSION)
        .max(ACP_PROTOCOL_VERSION);
    let mut agent_capabilities = json!({
        "loadSession": true,
        "promptCapabilities": {
            "image": true,
            "audio": false,
            "embeddedContext": true,
        },
        "mcpCapabilities": {
            "http": false,
            "sse": false,
        },
        "sessionCapabilities": {
            "close": {},
            "resume": {},
        }
    });

    if profile.is_extended()
        && let Some(object) = agent_capabilities.as_object_mut()
    {
        object.insert(
            "_meta".to_string(),
            json!({
                "jcode": {
                    "profile": profile.as_str(),
                    "extensions": ["raw_server_event"],
                    "capabilityProbeMethod": "session/capabilities",
                    "capabilities": {
                        "autocomplete": true,
                        "runSkill": false,
                        "applyPatch": false,
                        "memory": false,
                        "mcp": false
                    }
                }
            }),
        );
    }

    json!({
        "protocolVersion": protocol_version,
        "agentCapabilities": agent_capabilities,
        "agentInfo": {
            "name": "jcode",
            "title": "Jcode",
            "version": jcode_build_meta::PKG_VERSION,
        },
        "authMethods": [],
    })
}

fn build_effective_capabilities_snapshot(
    profile: AcpProfile,
    provider_choice: &ProviderChoice,
    configured_model: Option<&str>,
    provider_profile: Option<&str>,
    session_id: &str,
    provider_result: std::result::Result<Arc<dyn Provider>, String>,
) -> Value {
    let requested_provider = provider_choice.as_arg_value();
    let requested_model = configured_model.map(str::to_string);
    let profile_name = provider_profile
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    match provider_result {
        Ok(provider) => {
            let effective_model = provider.model();
            let effective_provider = provider.name().to_string();
            let context_window = provider.context_window();
            let transport = provider.transport();
            let auth_method = provider.active_auth_method_label().map(str::to_string);
            let tool_calling = profile.is_extended() && provider.handles_tools_internally();
            let provider_ready = !effective_model.trim().is_empty();
            json!({
                "sessionId": session_id,
                "variant": "fusion-forge-acp-v1",
                "provider": {
                    "requested": requested_provider,
                    "effective": effective_provider,
                    "displayName": provider.display_name(),
                    "profile": profile_name,
                    "transport": transport,
                    "authMethod": auth_method,
                    "type": provider_type_label(provider_choice),
                    "ready": provider_ready,
                },
                "model": {
                    "requested": requested_model,
                    "effective": effective_model,
                    "contextWindow": context_window,
                },
                "capabilities": {
                    "autocomplete": {
                        "available": profile.is_extended(),
                        "mode": "fim",
                        "notes": "Translated by Jcode provider adapters from universal prefix/suffix ACP input.",
                        "maxContextTokens": context_window,
                        "timeoutMsMax": 10_000,
                    },
                    "streaming": {
                        "available": true,
                    },
                    "toolCalling": {
                        "available": tool_calling,
                        "mode": if tool_calling { "native" } else { "unavailable" },
                    },
                    "skills": {
                        "available": false,
                        "requiresToolCalling": true,
                        "reason": if tool_calling {
                            "ACP skills contract is specified but not implemented in this runtime build."
                        } else {
                            "Active provider does not advertise native tool handling and ACP skills fallback is not implemented in this runtime build."
                        },
                    },
                    "patch": {
                        "available": false,
                        "preview": false,
                        "apply": false,
                        "reason": "ACP patch flow is not implemented in this runtime build.",
                    },
                    "memory": {
                        "available": false,
                        "list": false,
                        "forget": false,
                        "export": false,
                        "reason": "ACP memory surface is not implemented in this runtime build.",
                    },
                    "mcp": {
                        "available": false,
                        "status": false,
                        "reason": "ACP MCP status surface is not implemented in this runtime build.",
                    },
                }
            })
        }
        Err(error) => json!({
            "sessionId": session_id,
            "variant": "fusion-forge-acp-v1",
            "provider": {
                "requested": requested_provider,
                "effective": Value::Null,
                "displayName": Value::Null,
                "profile": profile_name,
                "transport": Value::Null,
                "authMethod": Value::Null,
                "type": provider_type_label(provider_choice),
                "ready": false,
                "reason": error,
            },
            "model": {
                "requested": requested_model,
                "effective": Value::Null,
                "contextWindow": Value::Null,
            },
            "capabilities": {
                "autocomplete": {
                    "available": false,
                    "mode": "fim",
                    "reason": "Provider initialization failed; autocomplete adapter unavailable.",
                    "timeoutMsMax": 10_000,
                },
                "streaming": {
                    "available": false,
                    "reason": "Provider initialization failed.",
                },
                "toolCalling": {
                    "available": false,
                    "reason": "Provider initialization failed.",
                },
                "skills": {
                    "available": false,
                    "reason": "Provider initialization failed.",
                },
                "patch": {
                    "available": false,
                    "preview": false,
                    "apply": false,
                    "reason": "Provider initialization failed.",
                },
                "memory": {
                    "available": false,
                    "list": false,
                    "forget": false,
                    "export": false,
                    "reason": "ACP memory surface is not implemented in this runtime build.",
                },
                "mcp": {
                    "available": false,
                    "status": false,
                    "reason": "ACP MCP status surface is not implemented in this runtime build.",
                },
            }
        }),
    }
}

fn provider_type_label(choice: &ProviderChoice) -> &'static str {
    match choice {
        ProviderChoice::Ollama | ProviderChoice::Lmstudio => "local",
        ProviderChoice::OpenaiCompatible => "custom",
        ProviderChoice::Jcode
        | ProviderChoice::Claude
        | ProviderChoice::AnthropicApi
        | ProviderChoice::Openai
        | ProviderChoice::OpenaiApi
        | ProviderChoice::Openrouter
        | ProviderChoice::Bedrock
        | ProviderChoice::Azure
        | ProviderChoice::Opencode
        | ProviderChoice::OpencodeGo
        | ProviderChoice::Zai
        | ProviderChoice::Kimi
        | ProviderChoice::Ai302
        | ProviderChoice::Baseten
        | ProviderChoice::Cortecs
        | ProviderChoice::Comtegra
        | ProviderChoice::Deepseek
        | ProviderChoice::Fpt
        | ProviderChoice::Firmware
        | ProviderChoice::HuggingFace
        | ProviderChoice::MoonshotAi
        | ProviderChoice::Nebius
        | ProviderChoice::Scaleway
        | ProviderChoice::Stackit
        | ProviderChoice::Groq
        | ProviderChoice::Mistral
        | ProviderChoice::Perplexity
        | ProviderChoice::TogetherAi
        | ProviderChoice::Deepinfra
        | ProviderChoice::Fireworks
        | ProviderChoice::Minimax
        | ProviderChoice::Xai
        | ProviderChoice::NvidiaNim
        | ProviderChoice::XiaomiMimo
        | ProviderChoice::Chutes
        | ProviderChoice::Cerebras
        | ProviderChoice::AlibabaCodingPlan
        | ProviderChoice::Cursor
        | ProviderChoice::Copilot
        | ProviderChoice::Gemini
        | ProviderChoice::Antigravity
        | ProviderChoice::Google
        | ProviderChoice::ClaudeSubprocess
        | ProviderChoice::Auto => "cloud",
    }
}

fn cwd_from_params(params: &Value) -> std::result::Result<PathBuf, String> {
    let cwd = match params.get("cwd").and_then(Value::as_str) {
        Some(cwd) if !cwd.trim().is_empty() => PathBuf::from(cwd),
        _ => std::env::current_dir().map_err(|err| err.to_string())?,
    };
    if !cwd.is_absolute() {
        return Err(format!("ACP cwd must be absolute: {}", cwd.display()));
    }
    Ok(cwd)
}

fn required_session_id(params: &Value) -> std::result::Result<String, String> {
    params
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "Missing required sessionId".to_string())
}

fn ensure_no_acp_mcp_servers(params: &Value) -> std::result::Result<(), String> {
    match params.get("mcpServers") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(items)) if items.is_empty() => Ok(()),
        Some(_) => Err(
            "ACP mcpServers are not supported yet; configure MCP servers in Jcode config.toml"
                .to_string(),
        ),
    }
}

fn prompt_from_params(
    params: &Value,
) -> std::result::Result<(String, Vec<(String, String)>), String> {
    let prompt = params
        .get("prompt")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing required prompt array".to_string())?;
    let mut text_parts = Vec::new();
    let mut images = Vec::new();

    for block in prompt {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("image") => {
                let mime_type = block
                    .get("mimeType")
                    .or_else(|| block.get("mime_type"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Image content block missing mimeType".to_string())?;
                let data = block
                    .get("data")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Image content block missing data".to_string())?;
                images.push((mime_type.to_string(), data.to_string()));
            }
            Some("resource") => {
                if let Some(resource) = block.get("resource") {
                    text_parts.push(format_resource_block(resource));
                }
            }
            Some("resource_link") => {
                let uri = block.get("uri").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or(uri);
                text_parts.push(format!("[Resource link: {name} <{uri}>]"));
            }
            Some(other) => {
                return Err(format!(
                    "Unsupported ACP prompt content block type: {other}"
                ));
            }
            None => return Err("Prompt content block missing type".to_string()),
        }
    }

    Ok((text_parts.join("\n\n"), images))
}

fn format_resource_block(resource: &Value) -> String {
    let uri = resource
        .get("uri")
        .and_then(Value::as_str)
        .unwrap_or("resource");
    if let Some(text) = resource.get("text").and_then(Value::as_str) {
        format!("[Embedded resource: {uri}]\n{text}")
    } else if let Some(blob) = resource.get("blob").and_then(Value::as_str) {
        let mime = resource
            .get("mimeType")
            .or_else(|| resource.get("mime_type"))
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        format!(
            "[Embedded binary resource: {uri} ({mime}, {} base64 bytes)]",
            blob.len()
        )
    } else {
        format!("[Embedded resource: {uri}]")
    }
}

fn agent_message_chunk(text: String) -> Value {
    json!({
        "sessionUpdate": "agent_message_chunk",
        "content": {
            "type": "text",
            "text": text,
        }
    })
}

fn tool_title(name: &str) -> String {
    match name {
        "bash" => "Running shell command".to_string(),
        "read" => "Reading file".to_string(),
        "write" => "Writing file".to_string(),
        "edit" | "multiedit" | "patch" | "apply_patch" => "Editing files".to_string(),
        "agentgrep" | "grep" | "glob" | "ls" => "Searching workspace".to_string(),
        "webfetch" | "websearch" => "Fetching web content".to_string(),
        other => other.replace('_', " "),
    }
}

pub(crate) fn tool_kind(name: &str) -> &'static str {
    match name {
        "read" => "read",
        "write" | "edit" | "multiedit" | "patch" | "apply_patch" => "edit",
        "bash" | "bg" | "selfdev" => "execute",
        "agentgrep" | "grep" | "glob" | "ls" | "session_search" | "conversation_search" => "search",
        "webfetch" | "websearch" | "codesearch" => "fetch",
        _ => "other",
    }
}

pub(crate) async fn run_acp_command(
    provider_choice: ProviderChoice,
    model: Option<String>,
    provider_profile: Option<String>,
    explicit_tool_profile: bool,
) -> Result<()> {
    crate::env::set_var("JCODE_NON_INTERACTIVE", "1");
    let acp_config = crate::config::config().acp.clone();
    if !explicit_tool_profile {
        crate::env::set_var("JCODE_TOOL_PROFILE", acp_config.tool_profile.trim());
        crate::config::invalidate_config_cache();
    }
    let profile = AcpProfile::parse(&acp_config.profile);
    AcpRuntime::new(profile, provider_choice, model, provider_profile)
        .run()
        .await
}

#[doc(hidden)]
pub async fn run_autocomplete_request_for_tests(
    provider: Arc<dyn Provider>,
    params: Value,
) -> Result<Value> {
    let request = autocomplete_request_from_params(&params).map_err(|err| anyhow::anyhow!(err))?;
    run_autocomplete_request(provider, request)
        .await
        .map_err(|err| anyhow::anyhow!("{}: {}", err.common_code, err.message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::StreamEvent;
    use crate::provider::EventStream;
    use async_stream::stream;
    use std::path::Path;
    use std::time::Duration;

    struct StaticAutocompleteProvider {
        response: String,
    }

    struct StaticCapabilityProvider {
        provider_name: &'static str,
        display_name: &'static str,
        model: &'static str,
        context_window: usize,
        handles_tools: bool,
        transport: Option<&'static str>,
    }

    #[async_trait::async_trait]
    impl Provider for StaticAutocompleteProvider {
        async fn complete(
            &self,
            _messages: &[crate::message::Message],
            _tools: &[crate::message::ToolDefinition],
            _system: &str,
            _resume_session_id: Option<&str>,
        ) -> Result<EventStream> {
            let response = self.response.clone();
            Ok(Box::pin(stream! {
                yield Ok(StreamEvent::TextDelta(response));
                yield Ok(StreamEvent::MessageEnd { stop_reason: Some("end_turn".to_string()) });
            }))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn model(&self) -> String {
            "mock-autocomplete".to_string()
        }

        fn fork(&self) -> Arc<dyn Provider> {
            Arc::new(Self {
                response: self.response.clone(),
            })
        }
    }

    #[async_trait::async_trait]
    impl Provider for StaticCapabilityProvider {
        async fn complete(
            &self,
            _messages: &[crate::message::Message],
            _tools: &[crate::message::ToolDefinition],
            _system: &str,
            _resume_session_id: Option<&str>,
        ) -> Result<EventStream> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn name(&self) -> &str {
            self.provider_name
        }

        fn display_name(&self) -> String {
            self.display_name.to_string()
        }

        fn model(&self) -> String {
            self.model.to_string()
        }

        fn handles_tools_internally(&self) -> bool {
            self.handles_tools
        }

        fn transport(&self) -> Option<String> {
            self.transport.map(str::to_string)
        }

        fn context_window(&self) -> usize {
            self.context_window
        }

        fn fork(&self) -> Arc<dyn Provider> {
            Arc::new(Self {
                provider_name: self.provider_name,
                display_name: self.display_name,
                model: self.model,
                context_window: self.context_window,
                handles_tools: self.handles_tools,
                transport: self.transport,
            })
        }
    }

    struct SlowAutocompleteProvider {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl Provider for SlowAutocompleteProvider {
        async fn complete(
            &self,
            _messages: &[crate::message::Message],
            _tools: &[crate::message::ToolDefinition],
            _system: &str,
            _resume_session_id: Option<&str>,
        ) -> Result<EventStream> {
            let delay = self.delay;
            Ok(Box::pin(stream! {
                tokio::time::sleep(delay).await;
                yield Ok(StreamEvent::TextDelta("late".to_string()));
            }))
        }

        fn name(&self) -> &str {
            "slow"
        }

        fn model(&self) -> String {
            "slow-model".to_string()
        }

        fn fork(&self) -> Arc<dyn Provider> {
            Arc::new(Self { delay: self.delay })
        }
    }

    #[test]
    fn acp_tool_kind_maps_core_tools() {
        assert_eq!(tool_kind("read"), "read");
        assert_eq!(tool_kind("apply_patch"), "edit");
        assert_eq!(tool_kind("bash"), "execute");
        assert_eq!(tool_kind("agentgrep"), "search");
        assert_eq!(tool_kind("webfetch"), "fetch");
        assert_eq!(tool_kind("swarm"), "other");
    }

    #[test]
    fn json_rpc_parse_errors_use_standard_codes() {
        let (code, _) = JsonRpcMessage::parse("not json").unwrap_err();
        assert_eq!(code, JSONRPC_PARSE_ERROR);

        let (code, message) = JsonRpcMessage::parse(r#"{"method":"initialize"}"#).unwrap_err();
        assert_eq!(code, JSONRPC_INVALID_REQUEST);
        assert!(message.contains("jsonrpc"));
    }

    #[test]
    fn prompt_from_params_accepts_text_images_and_resources() {
        let params = json!({
            "sessionId": "s1",
            "prompt": [
                {"type": "text", "text": "hello"},
                {"type": "image", "mimeType": "image/png", "data": "abc"},
                {"type": "resource", "resource": {"uri": "file:///tmp/a.rs", "text": "fn main(){}"}},
                {"type": "resource_link", "uri": "file:///tmp/b.rs", "name": "b.rs"}
            ]
        });
        let (text, images) = prompt_from_params(&params).unwrap();
        assert!(text.contains("hello"));
        assert!(text.contains("Embedded resource: file:///tmp/a.rs"));
        assert!(text.contains("Resource link: b.rs"));
        assert_eq!(images, vec![("image/png".to_string(), "abc".to_string())]);
    }

    #[test]
    fn initialize_standard_omits_jcode_meta() {
        let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Standard);
        assert_eq!(result["protocolVersion"], 1);
        assert!(result["agentCapabilities"].get("_meta").is_none());
        assert_eq!(result["agentCapabilities"]["loadSession"], true);
    }

    #[test]
    fn initialize_full_advertises_jcode_extension_meta() {
        let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Full);
        assert_eq!(
            result["agentCapabilities"]["_meta"]["jcode"]["profile"],
            "full"
        );
        assert_eq!(
            result["agentCapabilities"]["_meta"]["jcode"]["capabilities"]["autocomplete"],
            true
        );
        assert_eq!(
            result["agentCapabilities"]["_meta"]["jcode"]["capabilityProbeMethod"],
            "session/capabilities"
        );
    }

    #[test]
    fn event_mapper_maps_tool_lifecycle() {
        let mut mapper = EventMapper::new("session1".to_string(), AcpProfile::Standard);
        let start = mapper.map_event(ServerEvent::ToolStart {
            id: "tool1".to_string(),
            name: "bash".to_string(),
        });
        assert_eq!(start[0]["sessionUpdate"], "tool_call");
        assert_eq!(start[0]["kind"], "execute");

        let input = mapper.map_event(ServerEvent::ToolInput {
            delta: "{\"command\":\"true\"}".to_string(),
        });
        assert_eq!(input[0]["rawInput"]["command"], "true");

        let done = mapper.map_event(ServerEvent::ToolDone {
            id: "tool1".to_string(),
            name: "bash".to_string(),
            output: "ok".to_string(),
            error: None,
        });
        assert_eq!(done[0]["status"], "completed");
        assert_eq!(done[0]["content"][0]["content"]["text"], "ok");
    }

    #[test]
    fn non_empty_mcp_servers_rejected_until_session_scoped_mcp_is_supported() {
        let params = json!({"mcpServers": [{"name": "fs"}]});
        assert!(ensure_no_acp_mcp_servers(&params).is_err());
        let params = json!({"mcpServers": []});
        assert!(ensure_no_acp_mcp_servers(&params).is_ok());
    }

    #[test]
    fn cwd_must_be_absolute() {
        let params = json!({"cwd": "relative"});
        assert!(cwd_from_params(&params).is_err());
        let params = json!({"cwd": "/tmp"});
        assert_eq!(cwd_from_params(&params).unwrap(), Path::new("/tmp"));
    }

    #[test]
    fn autocomplete_request_rejects_inputs_over_limit() {
        let params = json!({
            "sessionId": "s1",
            "document": {
                "uri": "file:///tmp/app.ts",
                "languageId": "typescript",
                "version": 3,
                "prefix": "abcdef",
                "suffix": ""
            },
            "cursor": { "line": 0, "character": 6 },
            "limits": {
                "maxPrefixChars": 3,
                "maxSuffixChars": 0,
                "timeoutMs": 1000
            }
        });
        let error = autocomplete_request_from_params(&params).unwrap_err();
        assert!(error.contains("exceeds limits"));
    }

    #[test]
    fn normalize_autocomplete_completion_strips_fences_and_suffix_overlap() {
        let normalized = normalize_autocomplete_completion("```ts\ngetUser()\n}\n```", "}\n");
        assert_eq!(normalized, "getUser()\n");
    }

    #[tokio::test]
    async fn autocomplete_runtime_returns_structured_completion() {
        let provider: Arc<dyn Provider> = Arc::new(StaticAutocompleteProvider {
            response: "getUserById()\n}\n".to_string(),
        });
        let response = run_autocomplete_request(
            provider,
            autocomplete_request_from_params(&json!({
                "sessionId": "s1",
                "document": {
                    "uri": "file:///tmp/app.ts",
                    "languageId": "typescript",
                    "version": 7,
                    "prefix": "export function gre",
                    "suffix": "}\n"
                },
                "cursor": { "line": 0, "character": 19 },
                "limits": {
                    "maxPrefixChars": 4000,
                    "maxSuffixChars": 1000,
                    "timeoutMs": 1000
                }
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response["completion"], "getUserById()\n");
        assert_eq!(response["finishReason"], "completed");
        assert_eq!(response["providerEffective"]["providerName"], "mock");
    }

    #[tokio::test]
    async fn autocomplete_runtime_maps_timeout_to_common_error_code() {
        let provider: Arc<dyn Provider> = Arc::new(SlowAutocompleteProvider {
            delay: Duration::from_millis(25),
        });
        let error = run_autocomplete_request(
            provider,
            autocomplete_request_from_params(&json!({
                "sessionId": "s1",
                "document": {
                    "uri": "file:///tmp/app.ts",
                    "languageId": "typescript",
                    "version": 7,
                    "prefix": "export function gre",
                    "suffix": ""
                },
                "cursor": { "line": 0, "character": 19 },
                "limits": {
                    "maxPrefixChars": 4000,
                    "maxSuffixChars": 1000,
                    "timeoutMs": 1
                }
            }))
            .unwrap(),
        )
        .await
        .unwrap_err();

        assert_eq!(error.common_code, "AUTOCOMPLETE_TIMEOUT");
        assert!(error.retryable);
    }

    #[test]
    fn capability_snapshot_reports_effective_provider_state() {
        let snapshot = build_effective_capabilities_snapshot(
            AcpProfile::Extended,
            &ProviderChoice::Ollama,
            Some("llama3.2"),
            None,
            "session_123",
            Ok(Arc::new(StaticCapabilityProvider {
                provider_name: "ollama",
                display_name: "Ollama",
                model: "llama3.2",
                context_window: 32_768,
                handles_tools: false,
                transport: Some("http"),
            }) as Arc<dyn Provider>),
        );

        assert_eq!(snapshot["provider"]["requested"], "ollama");
        assert_eq!(snapshot["provider"]["effective"], "ollama");
        assert_eq!(snapshot["provider"]["type"], "local");
        assert_eq!(snapshot["model"]["contextWindow"], 32768);
        assert_eq!(snapshot["capabilities"]["autocomplete"]["available"], true);
        assert_eq!(snapshot["capabilities"]["toolCalling"]["available"], false);
        assert_eq!(snapshot["capabilities"]["skills"]["available"], false);
    }

    #[test]
    fn capability_snapshot_reports_provider_init_failure_without_throwing() {
        let snapshot = build_effective_capabilities_snapshot(
            AcpProfile::Extended,
            &ProviderChoice::Openai,
            Some("gpt-5.4"),
            None,
            "session_456",
            Err("missing credentials".to_string()),
        );

        assert_eq!(snapshot["provider"]["ready"], false);
        assert_eq!(snapshot["provider"]["requested"], "openai");
        assert_eq!(snapshot["capabilities"]["autocomplete"]["available"], false);
        assert_eq!(snapshot["capabilities"]["streaming"]["available"], false);
    }
}
