use crate::mock_provider::MockProvider;
use anyhow::Result;
use jcode::cli::acp::run_autocomplete_request_for_tests;
use jcode::message::StreamEvent;
use jcode::provider::Provider;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn acp_autocomplete_returns_completion_from_provider() -> Result<()> {
    let provider = MockProvider::new();
    provider.queue_response(vec![
        StreamEvent::TextDelta("etUserById".to_string()),
        StreamEvent::MessageEnd {
            stop_reason: Some("end_turn".to_string()),
        },
    ]);

    let response = run_autocomplete_request_for_tests(
        Arc::new(provider) as Arc<dyn Provider>,
        json!({
            "sessionId": "session_123",
            "document": {
                "uri": "file:///workspace/src/app.ts",
                "languageId": "typescript",
                "version": 12,
                "prefix": "export function gre",
                "suffix": "() {\n  return 42\n}\n"
            },
            "cursor": {
                "line": 0,
                "character": 19
            },
            "limits": {
                "maxPrefixChars": 4000,
                "maxSuffixChars": 1000,
                "timeoutMs": 1200
            }
        }),
    )
    .await?;

    assert_eq!(response["completion"], "etUserById");
    assert_eq!(response["providerEffective"]["providerName"], "mock");
    assert_eq!(response["finishReason"], "completed");
    Ok(())
}
