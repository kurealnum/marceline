//! Sentence-chunking of streamed LLM tokens into TTS (SPEC.md §5.3, EPIC 5.3).
//!
//! [`TtsEngine::synthesize`][super::TtsEngine::synthesize] takes an
//! already-segmented [`TextStream`], and sentence-chunking is explicitly
//! the *caller's* job (§2.4.1) so Kokoro-vs-Piper granularity differences
//! never leak into the trait. This module is that caller-side chunker: it
//! consumes an LLM [`ChatEventStream`], picks out [`ChatEvent::TextDelta`]
//! text, and emits one [`TextStream`] item per completed sentence.
//!
//! Emitting as soon as a sentence completes — rather than buffering the
//! whole answer — is what lets THINKING transition to SPEAKING on the
//! first sentence instead of waiting for the model to finish (§2.5, the
//! ≤1.5s wake→first-audio target of §9.2/§10).

use futures::{Stream, StreamExt};

use super::TextStream;
use crate::engine::EngineError;
use crate::llm::{ChatEvent, ChatEventStream};

/// Terminal punctuation that ends a sentence.
const SENTENCE_END: [char; 3] = ['.', '!', '?'];

/// Sentence-chunks a [`ChatEventStream`] into a [`TextStream`] for TTS.
///
/// - Only [`ChatEvent::TextDelta`] contributes text; `ToolCall*` events
///   belong to the tool broker (EPIC 6) and are silently skipped here —
///   they carry no text a listener should hear.
/// - A segment is emitted once terminal punctuation (`.`, `!`, `?`) is
///   followed by whitespace already in the buffer, so `"3.14"` mid-token
///   is not mistaken for a sentence boundary; a lone trailing punctuation
///   mark waits for either more text or [`ChatEvent::Done`].
/// - [`ChatEvent::Done`] flushes any trailing partial sentence, so the
///   last words of an answer are never dropped.
/// - Errors on the input stream propagate in-band as the corresponding
///   item on the output stream (invariant 1, §2.4.1), ending the chunker.
pub fn sentence_chunk(events: ChatEventStream) -> TextStream {
    Box::pin(SentenceChunker {
        events,
        buffer: String::new(),
        pending: std::collections::VecDeque::new(),
        finished: false,
    })
}

/// Stream adapter holding the chunker's accumulation state.
///
/// A hand-written [`Stream`] rather than `futures::stream::unfold`: a
/// single input event can complete more than one sentence (a burst
/// containing `"Hi. Bye."` in one `TextDelta`), so segments already found
/// are queued in `pending` and drained before the next poll of `events`.
struct SentenceChunker {
    events: ChatEventStream,
    /// Text accumulated since the last emitted sentence.
    buffer: String,
    /// Complete sentences found but not yet yielded.
    pending: std::collections::VecDeque<String>,
    /// Set once the input stream has ended, errored, or reported `Done`.
    finished: bool,
}

impl Stream for SentenceChunker {
    type Item = Result<String, EngineError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            if let Some(segment) = self.pending.pop_front() {
                return std::task::Poll::Ready(Some(Ok(segment)));
            }
            if self.finished {
                return std::task::Poll::Ready(None);
            }

            match self.events.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(Ok(event))) => match event {
                    ChatEvent::TextDelta(text) => {
                        self.buffer.push_str(&text);
                        self.drain_complete_sentences();
                        // A delta might not have completed a sentence;
                        // loop back to poll for more rather than
                        // returning `Poll::Ready(None)`-shaped nothing.
                        continue;
                    }
                    ChatEvent::ToolCallDelta { .. } | ChatEvent::ToolCallDone { .. } => continue,
                    ChatEvent::Done { .. } => {
                        self.finished = true;
                        self.flush_trailing();
                        continue;
                    }
                },
                std::task::Poll::Ready(Some(Err(err))) => {
                    self.finished = true;
                    return std::task::Poll::Ready(Some(Err(err)));
                }
                std::task::Poll::Ready(None) => {
                    self.finished = true;
                    self.flush_trailing();
                    continue;
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

impl SentenceChunker {
    /// Splits `buffer` on every sentence boundary it currently contains,
    /// queuing each completed sentence onto `pending`.
    ///
    /// A boundary is terminal punctuation immediately followed by
    /// whitespace that is already in the buffer — a mark at the very end
    /// of what has arrived so far is left in place, since the next token
    /// might turn `"3."` into `"3.14"` rather than a new sentence.
    fn drain_complete_sentences(&mut self) {
        loop {
            let Some(boundary) = self.find_boundary() else {
                return;
            };
            // `boundary` is a byte index of the punctuation mark; the
            // sentence includes it, the split point is the character
            // after it.
            let split_at = self.buffer[boundary..]
                .chars()
                .next()
                .map(|c| boundary + c.len_utf8())
                .unwrap_or(self.buffer.len());

            let rest = self.buffer.split_off(split_at);
            let segment = std::mem::replace(&mut self.buffer, rest);
            let trimmed = segment.trim();
            if !trimmed.is_empty() {
                self.pending.push_back(trimmed.to_string());
            }
        }
    }

    /// Byte index of the first sentence-ending punctuation mark that is
    /// followed by whitespace already present in the buffer.
    fn find_boundary(&self) -> Option<usize> {
        let mut chars = self.buffer.char_indices().peekable();
        while let Some((idx, ch)) = chars.next() {
            if SENTENCE_END.contains(&ch) {
                if let Some((_, next)) = chars.peek() {
                    if next.is_whitespace() {
                        return Some(idx);
                    }
                }
            }
        }
        None
    }

    /// Emits whatever is left in `buffer` as a final segment, for
    /// [`ChatEvent::Done`] or the input stream ending on its own.
    fn flush_trailing(&mut self) {
        let trimmed = self.buffer.trim();
        if !trimmed.is_empty() {
            self.pending.push_back(trimmed.to_string());
        }
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn event_stream(events: Vec<ChatEvent>) -> ChatEventStream {
        Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
    }

    async fn collect_segments(stream: TextStream) -> Vec<String> {
        stream
            .map(|item| item.expect("no error expected"))
            .collect()
            .await
    }

    #[tokio::test]
    async fn emits_one_segment_per_completed_sentence() {
        let events = event_stream(vec![
            ChatEvent::TextDelta("Hello there. How are you? ".to_string()),
            ChatEvent::TextDelta("Fine!".to_string()),
            ChatEvent::Done {
                finish_reason: crate::llm::FinishReason::Stop,
            },
        ]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(
            segments,
            vec![
                "Hello there.".to_string(),
                "How are you?".to_string(),
                "Fine!".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn a_burst_containing_multiple_sentences_splits_into_each() {
        let events = event_stream(vec![
            ChatEvent::TextDelta("Hi. Bye. ".to_string()),
            ChatEvent::Done {
                finish_reason: crate::llm::FinishReason::Stop,
            },
        ]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(segments, vec!["Hi.".to_string(), "Bye.".to_string()]);
    }

    #[tokio::test]
    async fn a_decimal_point_mid_token_is_not_a_sentence_boundary() {
        let events = event_stream(vec![
            ChatEvent::TextDelta("Pi is about 3.".to_string()),
            ChatEvent::TextDelta("14 today.".to_string()),
            ChatEvent::Done {
                finish_reason: crate::llm::FinishReason::Stop,
            },
        ]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(segments, vec!["Pi is about 3.14 today.".to_string()]);
    }

    #[tokio::test]
    async fn trailing_partial_sentence_is_flushed_on_done() {
        let events = event_stream(vec![
            ChatEvent::TextDelta("No terminal punctuation here".to_string()),
            ChatEvent::Done {
                finish_reason: crate::llm::FinishReason::Stop,
            },
        ]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(segments, vec!["No terminal punctuation here".to_string()]);
    }

    #[tokio::test]
    async fn trailing_partial_sentence_is_flushed_when_the_stream_just_ends() {
        // No explicit `Done` — the stream simply has no more items.
        let events = event_stream(vec![ChatEvent::TextDelta("unfinished".to_string())]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(segments, vec!["unfinished".to_string()]);
    }

    #[tokio::test]
    async fn tool_call_events_are_skipped_and_carry_no_text() {
        let events = event_stream(vec![
            ChatEvent::TextDelta("Checking the weather.".to_string()),
            ChatEvent::ToolCallDelta {
                id: "call-1".to_string(),
                name: Some("get_weather".to_string()),
                args_delta: "{\"city\":".to_string(),
            },
            ChatEvent::ToolCallDelta {
                id: "call-1".to_string(),
                name: None,
                args_delta: "\"nyc\"}".to_string(),
            },
            ChatEvent::ToolCallDone {
                id: "call-1".to_string(),
            },
            ChatEvent::TextDelta(" Done.".to_string()),
            ChatEvent::Done {
                finish_reason: crate::llm::FinishReason::ToolCalls,
            },
        ]);

        let segments = collect_segments(sentence_chunk(events)).await;

        assert_eq!(
            segments,
            vec!["Checking the weather.".to_string(), "Done.".to_string()]
        );
    }

    #[tokio::test]
    async fn an_upstream_error_propagates_in_band_and_ends_the_stream() {
        let events: ChatEventStream = Box::pin(futures::stream::iter(vec![
            Ok(ChatEvent::TextDelta("Partial sentence".to_string())),
            Err(EngineError::Worker {
                backend: "llm",
                message: "connection reset".to_string(),
            }),
        ]));

        let mut segments = sentence_chunk(events);
        let items: Vec<_> = segments.by_ref().collect().await;

        assert_eq!(items.len(), 1);
        let err = items[0].as_ref().expect_err("expected an in-band error");
        assert!(matches!(err, EngineError::Worker { .. }));
    }

    #[tokio::test]
    async fn first_sentence_is_emitted_before_the_stream_completes() {
        // A slow token stream: the second delta never arrives during this
        // test, so a chunker that waited for `Done` before emitting
        // anything would hang here instead of producing the first
        // sentence promptly.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<ChatEvent, EngineError>>();
        let events: ChatEventStream = Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx));

        tx.send(Ok(ChatEvent::TextDelta(
            "First sentence. ".to_string(),
        )))
        .unwrap();

        let mut stream = sentence_chunk(events);
        let first = tokio::time::timeout(Duration::from_millis(200), stream.next())
            .await
            .expect("first sentence should arrive without waiting for the rest of the stream")
            .expect("expected a segment")
            .expect("no error expected");

        assert_eq!(first, "First sentence.");

        // Clean up: finish the stream so the test does not leak the sender.
        tx.send(Ok(ChatEvent::Done {
            finish_reason: crate::llm::FinishReason::Stop,
        }))
        .unwrap();
        drop(tx);
    }
}
