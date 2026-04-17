use crate::{
    framing::FramingParserSection,
    reduction::{ReductionParserError, ReductionParserGroup, ReductionParserSection, ReductionParserState},
    types::Token,
};

/// Defines how incoming text is captured at the current stack level
#[derive(Debug)]
enum ReductionParserCaptureMode {
    /// All greedy children completed (or none defined)
    Direct,
    /// Capturing into a greedy child at `child_index`
    Greedy {
        child_index: usize,
        items_captured: usize,
    },
}

/// Defines one level of group nesting on the stack
#[derive(Debug)]
struct ReductionParserStackEntry {
    /// Index of this group in parent's sections vec
    section_index: usize,
    /// The group definition this entry was created from
    definition: ReductionParserGroup,
    /// How text is currently being captured inside this group
    capture_mode: ReductionParserCaptureMode,
}

impl ReductionParserStackEntry {
    fn child_groups(&self) -> &[ReductionParserGroup] {
        self.definition.groups()
    }

    /// Count of leading greedy children in this entry's child groups
    fn greedy_child_count(&self) -> usize {
        self.child_groups().iter().take_while(|group| matches!(group, ReductionParserGroup::Greedy { .. })).count()
    }

    /// Whether this token matches a close token for this entry
    fn matches_close_token(
        &self,
        token_value: &str,
    ) -> bool {
        match &self.definition {
            ReductionParserGroup::Bounded {
                close_tokens,
                ..
            } => close_tokens.iter().any(|token| token == token_value),
            ReductionParserGroup::Open {
                ..
            } => false,
            ReductionParserGroup::Greedy {
                ..
            } => false,
        }
    }

    /// Whether this entry auto-closes when a sibling's open token appears
    fn closes_on_sibling(&self) -> bool {
        match &self.definition {
            ReductionParserGroup::Bounded {
                ..
            } => false,
            ReductionParserGroup::Open {
                ..
            } => true,
            ReductionParserGroup::Greedy {
                capturing_limit,
                ..
            } => capturing_limit.is_none(),
        }
    }

    /// Capture limit of the current greedy child, if any
    fn capture_limit(&self) -> Option<usize> {
        match &self.capture_mode {
            ReductionParserCaptureMode::Greedy {
                child_index,
                ..
            } => self.child_groups()[*child_index].capturing_limit(),
            ReductionParserCaptureMode::Direct => None,
        }
    }

    /// Whether the current greedy child has reached its capture limit
    fn is_capture_complete(&self) -> bool {
        match &self.capture_mode {
            ReductionParserCaptureMode::Greedy {
                items_captured,
                ..
            } => self.capture_limit().map(|limit| *items_captured >= limit).unwrap_or(false),
            ReductionParserCaptureMode::Direct => false,
        }
    }

    /// Record captured items in the current greedy child
    fn increment_capture(
        &mut self,
        count: usize,
    ) {
        if let ReductionParserCaptureMode::Greedy {
            items_captured,
            ..
        } = &mut self.capture_mode
        {
            *items_captured += count;
        }
    }

    /// Close current greedy child, advance to next or switch to Direct mode
    fn advance_greedy(&mut self) {
        if let ReductionParserCaptureMode::Greedy {
            child_index,
            ..
        } = &self.capture_mode
        {
            let next_index = child_index + 1;
            if next_index < self.greedy_child_count() {
                self.capture_mode = ReductionParserCaptureMode::Greedy {
                    child_index: next_index,
                    items_captured: 0,
                };
            } else {
                self.capture_mode = ReductionParserCaptureMode::Direct;
            }
        }
    }
}

pub(crate) struct ReductionParserGroupStack {
    entries: Vec<ReductionParserStackEntry>,
    root_groups: Vec<ReductionParserGroup>,
}

impl ReductionParserGroupStack {
    pub fn new(root_groups: Vec<ReductionParserGroup>) -> Self {
        Self {
            entries: Vec::new(),
            root_groups,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// Token matching

impl ReductionParserGroupStack {
    /// Is this token a close token for any ancestor? If yes, close all groups down to that ancestor
    #[tracing::instrument(skip_all, fields(token = %token))]
    pub fn close_groups_matching_token(
        &mut self,
        token: &Token,
        state: &mut ReductionParserState,
    ) -> Result<bool, ReductionParserError> {
        let Some(depth) = self.find_matching_close_depth(&token.value) else {
            return Ok(false);
        };
        self.close_groups_to_depth(depth, Some(token.clone()), state)?;
        self.advance_completed_greedy_captures(state)?;
        Ok(true)
    }

    /// Is this token an open token? Close any greedy that yields to siblings, then check child groups and root groups
    #[tracing::instrument(skip_all, fields(token = %token))]
    pub fn open_group_matching_token(
        &mut self,
        token: &Token,
        state: &mut ReductionParserState,
    ) -> Result<bool, ReductionParserError> {
        self.close_greedy_on_sibling_token(&token.value, state)?;

        if let Some(group) = self.find_matching_group(&token.value) {
            self.open_group(group, Some(token.clone()), state)?;
            return Ok(true);
        }

        if let Some(group) = self.find_matching_root_group(&token.value) {
            self.open_group(group, Some(token.clone()), state)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Walk stack bottom-up, find which level closes on this token
    fn find_matching_close_depth(
        &self,
        token_value: &str,
    ) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find(|(_, entry)| entry.matches_close_token(token_value))
            .map(|(index, _)| index)
    }

    /// Does this token match an open token of any group available at the current level
    fn find_matching_group(
        &self,
        token_value: &str,
    ) -> Option<ReductionParserGroup> {
        let child_groups = match self.entries.last() {
            Some(entry) => entry.child_groups(),
            None => &self.root_groups,
        };
        child_groups.iter().find(|group| group.open_token().map(String::as_str) == Some(token_value)).cloned()
    }

    /// Does this token match an open token of a root-level group? (re-entry from nested)
    fn find_matching_root_group(
        &self,
        token_value: &str,
    ) -> Option<ReductionParserGroup> {
        if self.entries.is_empty() {
            return None;
        }
        self.root_groups.iter().find(|group| group.open_token().map(String::as_str) == Some(token_value)).cloned()
    }

    /// Does this token match any sibling at the current depth? (for greedy close detection)
    fn is_sibling_open_token(
        &self,
        token_value: &str,
    ) -> bool {
        let parent_children = if self.entries.len() <= 1 {
            &self.root_groups
        } else {
            self.entries[self.entries.len() - 2].child_groups()
        };
        parent_children.iter().any(|group| group.open_token().map(String::as_str) == Some(token_value))
    }
}

// Frame operations
impl ReductionParserGroupStack {
    /// If current level has an unopened greedy child, open it so it can receive content
    #[tracing::instrument(skip_all)]
    pub fn open_greedy_group_if_needed(
        &mut self,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let Some(entry) = self.entries.last() else {
            return Ok(());
        };

        let ReductionParserCaptureMode::Greedy {
            child_index,
            ..
        } = &entry.capture_mode
        else {
            return Ok(());
        };

        if *child_index >= entry.greedy_child_count() {
            return Ok(());
        }

        let child_definition = entry.child_groups()[*child_index].clone();
        self.open_group(child_definition, None, state)
    }

    /// Push a new text frame, record capture, check if greedy is full
    #[tracing::instrument(skip_all, fields(tokens = tokens.len()))]
    pub fn append_frame_text(
        &mut self,
        tokens: Vec<Token>,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let count = tokens.len();
        let sections = self.current_sections(state)?;
        sections.push(ReductionParserSection::Frame(FramingParserSection::Text(tokens)));
        self.record_captured_items(count);
        self.advance_completed_greedy_captures(state)
    }

    /// Push a new marker frame, record capture, check if greedy is full
    #[tracing::instrument(skip_all, fields(token = %token))]
    pub fn append_frame_marker(
        &mut self,
        token: Token,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let sections = self.current_sections(state)?;
        sections.push(ReductionParserSection::Frame(FramingParserSection::Marker(token)));
        self.record_captured_items(1);
        self.advance_completed_greedy_captures(state)
    }

    /// Append token to last text frame (or create new one), record capture, check if greedy is full
    #[tracing::instrument(skip_all, fields(token = %token))]
    pub fn extend_frame_text(
        &mut self,
        token: Token,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let sections = self.current_sections(state)?;
        match sections.last_mut() {
            Some(ReductionParserSection::Frame(FramingParserSection::Text(tokens))) => {
                tokens.push(token);
            },
            _ => {
                sections.push(ReductionParserSection::Frame(FramingParserSection::Text(vec![token])));
            },
        }
        self.record_captured_items(1);
        self.advance_completed_greedy_captures(state)
    }
}

// Lifecycle
impl ReductionParserGroupStack {
    /// Create section in state, push entry to stack with initial capture mode
    #[tracing::instrument(skip_all, fields(name = definition.name()))]
    fn open_group(
        &mut self,
        definition: ReductionParserGroup,
        open_token: Option<Token>,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let name = definition.name().to_string();
        let has_greedy_children =
            definition.groups().iter().any(|group| matches!(group, ReductionParserGroup::Greedy { .. }));
        let capture_mode = if has_greedy_children {
            ReductionParserCaptureMode::Greedy {
                child_index: 0,
                items_captured: 0,
            }
        } else {
            ReductionParserCaptureMode::Direct
        };

        let sections = self.current_sections(state)?;
        let index = sections.len();
        sections.push(ReductionParserSection::Group {
            name,
            open: open_token,
            close: None,
            finished: false,
            sections: Vec::new(),
        });

        self.entries.push(ReductionParserStackEntry {
            section_index: index,
            definition,
            capture_mode,
        });

        Ok(())
    }

    /// Pop stack, set close token on the section
    fn close_top_group(
        &mut self,
        close_token: Option<Token>,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        let Some(entry) = self.entries.pop() else {
            return Ok(());
        };
        let _span = tracing::info_span!("close_top_group", name = entry.definition.name()).entered();
        let sections = self.current_sections(state)?;
        if let Some(ReductionParserSection::Group {
            close,
            finished,
            ..
        }) = sections.get_mut(entry.section_index)
        {
            *close = close_token;
            *finished = true;
        }
        Ok(())
    }

    /// Close everything above target depth, then close the target with the close token
    #[tracing::instrument(skip_all, fields(depth = depth))]
    fn close_groups_to_depth(
        &mut self,
        depth: usize,
        close_token: Option<Token>,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        while self.entries.len() > depth + 1 {
            self.close_top_group(None, state)?;
        }
        self.close_top_group(close_token, state)
    }
}

// Greedy capture management
impl ReductionParserGroupStack {
    /// While the top entry is an unlimited greedy and the token is a sibling, keep closing
    #[tracing::instrument(skip_all, fields(token_value = token_value))]
    fn close_greedy_on_sibling_token(
        &mut self,
        token_value: &str,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        while let Some(entry) = self.entries.last() {
            if entry.closes_on_sibling() && self.is_sibling_open_token(token_value) {
                self.close_top_group(None, state)?;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Walk stack bottom-up, find nearest entry with a capture limit, increment its counter
    fn record_captured_items(
        &mut self,
        count: usize,
    ) {
        for entry in self.entries.iter_mut().rev() {
            if entry.capture_limit().is_some() {
                entry.increment_capture(count);
                return;
            }
        }
    }

    /// Find any entry that hit its capture limit, close children above it, advance to next greedy or Direct, open next greedy if needed
    #[tracing::instrument(skip_all)]
    fn advance_completed_greedy_captures(
        &mut self,
        state: &mut ReductionParserState,
    ) -> Result<(), ReductionParserError> {
        loop {
            let target_depth = self.entries.iter().enumerate().rev().find_map(|(index, entry)| {
                if entry.is_capture_complete() {
                    Some(index)
                } else {
                    None
                }
            });

            let Some(depth) = target_depth else {
                break;
            };

            while self.entries.len() > depth + 1 {
                self.close_top_group(None, state)?;
            }

            if let Some(entry) = self.entries.last_mut() {
                entry.advance_greedy();
            }

            self.open_greedy_group_if_needed(state)?;
        }
        Ok(())
    }
}

// Navigation
impl ReductionParserGroupStack {
    /// Walk the stack following section indices to get mutable reference to the current group's sections
    fn current_sections<'a>(
        &self,
        state: &'a mut ReductionParserState,
    ) -> Result<&'a mut Vec<ReductionParserSection>, ReductionParserError> {
        let mut current = &mut state.sections;
        for entry in &self.entries {
            match current.get_mut(entry.section_index) {
                Some(ReductionParserSection::Group {
                    sections,
                    ..
                }) => {
                    current = sections;
                },
                _ => {
                    return Err(ReductionParserError::InvalidState {
                        index: entry.section_index,
                    });
                },
            }
        }
        Ok(current)
    }
}
