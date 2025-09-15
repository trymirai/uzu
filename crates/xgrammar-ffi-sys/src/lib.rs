#![allow(non_camel_case_types)]
use std::ffi::c_void;

extern "C" {
    pub fn xgr_matcher_new_from_json(
        tokenizer_info_json: *const u8,
        tokenizer_info_json_len: usize,
        compiled_grammar_json: *const u8,
        compiled_grammar_json_len: usize,
    ) -> *mut c_void;

    pub fn xgr_matcher_free(m: *mut c_void);

    pub fn xgr_fill_next_token_bitmask_packed(
        m: *mut c_void,
        data: *mut u32,
        batch: i32,
        nwords32: i32,
        index: i32,
        debug_print: i32,
    ) -> i32;

    pub fn xgr_accept_token(
        m: *mut c_void,
        token_id: i32,
        debug_print: i32,
    ) -> i32;
}

pub type MatcherHandle = *mut c_void;
