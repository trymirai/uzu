use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use uzu_engine::{
    backends::metal::Metal,
    engine::language_model::LanguageModel,
    vision::{AnyVlmPipeline, hash_image, MultimodalMessage, VlmState},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMessageSummary {
    pub role: String,
    pub content: Option<String>,
    pub image_hashes: Vec<u64>,
    /// One fingerprint per video (first frame hash mixed with the frame count),
    /// so a video conversation never prefix-matches an image conversation.
    pub video_hashes: Vec<u64>,
}

pub struct VlmSession {
    #[allow(dead_code)]
    pub id: String,
    pub state: VlmState,
    pub history: Vec<SessionMessageSummary>,
    pub prompt_tokens_count: usize,
    pub last_accessed: Instant,
    /// Whether the most recent generation on this session ended in tool calls; a
    /// following tool-result tail can then be prefilled incrementally on top of it.
    pub last_generation_tool_call: bool,
}

pub struct VlmSessionManager {
    sessions: Mutex<HashMap<String, Arc<Mutex<VlmSession>>>>,
    max_sessions: usize,
    session_ttl: Duration,
}

impl VlmSessionManager {
    pub fn new(max_sessions: usize, session_ttl: Duration) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            max_sessions,
            session_ttl,
        }
    }

    /// Live sessions (including ones awaiting TTL eviction).
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }

    pub fn session_ttl(&self) -> Duration {
        self.session_ttl
    }

    /// Extract lightweight message summary (role, content, image hashes)
    pub fn summarize_messages(messages: &[MultimodalMessage]) -> Vec<SessionMessageSummary> {
        messages
            .iter()
            .map(|m| SessionMessageSummary {
                role: m.role.clone(),
                content: m.content.clone(),
                image_hashes: m.images.iter().map(hash_image).collect(),
                video_hashes: m
                    .videos
                    .iter()
                    .map(|frames| {
                        let first = frames.first().map(hash_image).unwrap_or(0);
                        first.rotate_left(17) ^ (frames.len() as u64)
                    })
                    .collect(),
            })
            .collect()
    }

    /// Find an existing session matching explicit ID or implicit prefix, or create a new session
    pub async fn get_or_create(
        &self,
        requested_session_id: Option<&str>,
        incoming_summary: &[SessionMessageSummary],
        vlm: &AnyVlmPipeline,
        model: &LanguageModel<Metal>,
        max_context_tokens: u32,
    ) -> Result<(Arc<Mutex<VlmSession>>, bool, String), anyhow::Error> {
        let mut guard = self.sessions.lock().await;
        let now = Instant::now();

        // 1. Evict expired sessions
        let mut expired = Vec::new();
        for (id, s) in guard.iter() {
            if let Ok(sess) = s.try_lock()
                && now.duration_since(sess.last_accessed) > self.session_ttl {
                    expired.push(id.clone());
                }
        }
        for id in expired {
            if let Some(s) = guard.remove(&id)
                && let Ok(sess) = Arc::try_unwrap(s) {
                    let sess = sess.into_inner();
                    vlm.release_state(sess.state);
                }
        }

        // 2. Check explicit session_id
        if let Some(sid) = requested_session_id
            && let Some(session) = guard.get(sid) {
                let mut s = session.lock().await;
                s.last_accessed = now;
                drop(s);
                return Ok((Arc::clone(session), false, sid.to_string()));
            }

        // 3. Implicit prefix matching across active sessions
        if !incoming_summary.is_empty() {
            for (id, session) in guard.iter() {
                let mut s = session.lock().await;
                if now.duration_since(s.last_accessed) <= self.session_ttl && !s.history.is_empty()
                    && incoming_summary.len() > s.history.len()
                        && incoming_summary[..s.history.len()] == s.history[..]
                    {
                        s.last_accessed = now;
                        let sid = id.clone();
                        drop(s);
                        return Ok((Arc::clone(session), false, sid));
                    }
            }
        }

        // 4. If at capacity, evict LRU session
        if guard.len() >= self.max_sessions {
            let mut oldest_id: Option<String> = None;
            let mut oldest_time = now;
            for (id, s) in guard.iter() {
                if let Ok(sess) = s.try_lock()
                    && sess.last_accessed < oldest_time {
                        oldest_time = sess.last_accessed;
                        oldest_id = Some(id.clone());
                    }
            }
            if let Some(id) = oldest_id
                && let Some(s) = guard.remove(&id)
                    && let Ok(sess) = Arc::try_unwrap(s) {
                        let sess = sess.into_inner();
                        vlm.release_state(sess.state);
                    }
        }

        // 5. Create new session with unique server-generated ID
        let new_id = requested_session_id
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("sess_{}", uuid::Uuid::new_v4().simple()));

        let state = vlm.acquire_state(model, Some(max_context_tokens))?;
        let session = Arc::new(Mutex::new(VlmSession {
            id: new_id.clone(),
            state,
            history: Vec::new(),
            prompt_tokens_count: 0,
            last_accessed: now,
            last_generation_tool_call: false,
        }));

        guard.insert(new_id.clone(), Arc::clone(&session));
        Ok((session, true, new_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_message_summary_matching() {
        let h1 = vec![
            SessionMessageSummary {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                image_hashes: vec![12345],
                video_hashes: Vec::new(),
            },
            SessionMessageSummary {
                role: "assistant".to_string(),
                content: Some("hi".to_string()),
                image_hashes: vec![],
                video_hashes: Vec::new(),
            },
        ];

        let mut incoming = h1.clone();
        assert_eq!(incoming[..h1.len()], h1[..]);

        // Follow-up message without images
        incoming.push(SessionMessageSummary {
            role: "user".to_string(),
            content: Some("how are you?".to_string()),
            image_hashes: vec![],
                video_hashes: Vec::new(),
        });

        assert!(incoming.len() > h1.len());
        assert_eq!(incoming[..h1.len()], h1[..]);
        assert!(incoming.last().unwrap().image_hashes.is_empty());
    }

    #[test]
    fn test_session_message_summary_mismatch() {
        let h1 = [SessionMessageSummary {
                role: "user".to_string(),
                content: Some("image 1".to_string()),
                image_hashes: vec![111],
                video_hashes: Vec::new(),
            }];

        let incoming = [SessionMessageSummary {
                role: "user".to_string(),
                content: Some("image 2".to_string()),
                image_hashes: vec![222],
                video_hashes: Vec::new(),
            }];

        assert_ne!(incoming[..h1.len()], h1[..]);
    }
}
