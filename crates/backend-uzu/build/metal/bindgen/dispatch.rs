use std::iter::repeat_n;

use anyhow::{Result, bail};
use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

use super::{
    super::ast::{MetalArgumentType, MetalGroupsType, MetalKernelInfo},
    variant_path_rewriter::VariantPathRewriter,
};

pub struct DispatchEmission {
    pub dispatch_code: TokenStream,
    pub empty_dispatch_guards: TokenStream,
}

pub fn parse(
    kernel: &MetalKernelInfo,
    variant_path_rewriter: &mut VariantPathRewriter,
) -> Result<DispatchEmission> {
    let (dispatch_code, dispatch_size_expressions) = if kernel.has_axis() {
        if kernel.has_groups() || kernel.has_threads() {
            bail!("mixing groups/threads and axis is not supported");
        }
        build_axis_dispatch(kernel, variant_path_rewriter)?
    } else if kernel.has_groups_indirect() {
        if kernel.has_groups_direct() {
            bail!("cannot mix indirect and direct groups");
        }
        build_indirect_dispatch(kernel, variant_path_rewriter)?
    } else {
        build_direct_dispatch(kernel, variant_path_rewriter)?
    };

    let empty_dispatch_guards = build_empty_dispatch_guards(&dispatch_size_expressions);

    Ok(DispatchEmission {
        dispatch_code,
        empty_dispatch_guards,
    })
}

fn build_axis_dispatch(
    kernel: &MetalKernelInfo,
    variant_path_rewriter: &mut VariantPathRewriter,
) -> Result<(TokenStream, Vec<TokenStream>)> {
    let mut axis_pairs: Vec<(TokenStream, TokenStream)> = kernel
        .arguments
        .iter()
        .filter_map(|argument| match argument.argument_type() {
            Ok(MetalArgumentType::Axis(threads_text, threads_per_group_text)) => {
                Some((threads_text, threads_per_group_text))
            },
            _ => None,
        })
        .map(|(threads_text, threads_per_group_text)| -> Result<(TokenStream, TokenStream)> {
            let threads = variant_path_rewriter.rewrite(&threads_text)?;
            let threads_per_group = variant_path_rewriter.rewrite(&threads_per_group_text)?;
            Ok((threads, threads_per_group))
        })
        .collect::<Result<_>>()?;
    axis_pairs.extend(repeat_n((quote! { 1 }, quote! { 1 }), 3 - axis_pairs.len()));

    let (threads, threads_per_group): (Vec<TokenStream>, Vec<TokenStream>) = axis_pairs.into_iter().unzip();

    let dispatch_code = quote! {
        compute_encoder.dispatch_threads(
            MTLSize::new(#((#threads) as usize, )*),
            MTLSize::new(#((#threads_per_group) as usize, )*),
        );
    };

    let mut dispatch_size_expressions = threads;
    dispatch_size_expressions.extend(threads_per_group);
    Ok((dispatch_code, dispatch_size_expressions))
}

fn build_indirect_dispatch(
    kernel: &MetalKernelInfo,
    variant_path_rewriter: &mut VariantPathRewriter,
) -> Result<(TokenStream, Vec<TokenStream>)> {
    let mut threads = collect_thread_expressions(kernel, variant_path_rewriter)?;
    threads.extend(repeat_n(quote! { 1 }, 3 - threads.len()));

    let dispatch_code = quote! {
        compute_encoder.dispatch_threadgroups_indirect(
            __dsl_indirect_dispatch_buffer.0,
            __dsl_indirect_dispatch_buffer.1,
            MTLSize::new(#((#threads) as usize, )*),
        );
    };

    Ok((dispatch_code, threads))
}

fn build_direct_dispatch(
    kernel: &MetalKernelInfo,
    variant_path_rewriter: &mut VariantPathRewriter,
) -> Result<(TokenStream, Vec<TokenStream>)> {
    let mut threads = collect_thread_expressions(kernel, variant_path_rewriter)?;
    threads.extend(repeat_n(quote! { 1 }, 3 - threads.len()));

    let mut groups: Vec<TokenStream> = kernel
        .arguments
        .iter()
        .filter_map(|argument| match argument.argument_type() {
            Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(groups_text))) => {
                Some(variant_path_rewriter.rewrite(&groups_text))
            },
            _ => None,
        })
        .collect::<Result<_>>()?;
    groups.extend(repeat_n(quote! { 1 }, 3 - groups.len()));

    let dispatch_code = quote! {
        compute_encoder.dispatch_threadgroups(
            MTLSize::new(#((#groups) as usize, )*),
            MTLSize::new(#((#threads) as usize, )*),
        );
    };

    let mut dispatch_size_expressions = groups;
    dispatch_size_expressions.extend(threads);
    Ok((dispatch_code, dispatch_size_expressions))
}

fn collect_thread_expressions(
    kernel: &MetalKernelInfo,
    variant_path_rewriter: &mut VariantPathRewriter,
) -> Result<Vec<TokenStream>> {
    kernel
        .arguments
        .iter()
        .filter_map(|argument| match argument.argument_type() {
            Ok(MetalArgumentType::Threads(threads_text)) => Some(variant_path_rewriter.rewrite(&threads_text)),
            _ => None,
        })
        .collect()
}

fn build_empty_dispatch_guards(dispatch_size_expressions: &[TokenStream]) -> TokenStream {
    let combined_guard = dispatch_size_expressions
        .iter()
        .filter(|expression| !is_positive_integer_literal(expression))
        .map(|expression| quote! { (#expression) == 0 })
        .reduce(|left, right| quote! { #left || #right });

    match combined_guard {
        Some(guard) => quote! { if #guard { return; }; },
        None => quote! {},
    }
}

fn is_positive_integer_literal(expression: &TokenStream) -> bool {
    syn::parse2::<LitInt>(expression.clone())
        .ok()
        .and_then(|literal| literal.base10_parse::<u32>().ok())
        .is_some_and(|value| value != 0)
}
