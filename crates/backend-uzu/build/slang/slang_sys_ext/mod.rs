macro_rules! vcall {
	($self:expr, $method:ident($($args:expr),*)) => {
		unsafe { ($self.vtable().$method)($self.as_raw(), $($args),*) }
	};
}

mod load_module_with_diagnostics;

pub use load_module_with_diagnostics::SessionLoadModuleWithDiagnosticsExt;
