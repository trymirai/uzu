use xgrammar_ffi_sys as ffi;

pub struct Matcher(pub(crate) ffi::MatcherHandle);

unsafe impl Send for Matcher {}
unsafe impl Sync for Matcher {}

impl Drop for Matcher {
    fn drop(&mut self) {
        unsafe { ffi::xgr_matcher_free(self.0) }
    }
}

impl Matcher {
    pub fn from_json(
        tokenizer_info_json: &str,
        compiled_grammar_json: &str,
    ) -> Option<Self> {
        let h = unsafe {
            ffi::xgr_matcher_new_from_json(
                tokenizer_info_json.as_ptr(),
                tokenizer_info_json.len(),
                compiled_grammar_json.as_ptr(),
                compiled_grammar_json.len(),
            )
        };
        if h.is_null() {
            None
        } else {
            Some(Self(h))
        }
    }

    pub fn fill_next_token_mask_row(
        &mut self,
        out_words: &mut [u32],
        batch: usize,
        nwords32: usize,
        row: usize,
    ) -> bool {
        let ok = unsafe {
            ffi::xgr_fill_next_token_bitmask_packed(
                self.0,
                out_words.as_mut_ptr(),
                batch as i32,
                nwords32 as i32,
                row as i32,
                0,
            )
        };
        ok == 1
    }

    pub fn accept(
        &mut self,
        token_id: i32,
    ) -> bool {
        unsafe { ffi::xgr_accept_token(self.0, token_id, 0) == 1 }
    }
}
