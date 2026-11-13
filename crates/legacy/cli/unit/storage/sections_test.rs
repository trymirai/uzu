use super::*;

fn counts(
    installed: usize,
    downloading: usize,
    available: usize,
) -> impl Fn(Section) -> usize {
    move |section| match section {
        Section::Installed => installed,
        Section::Downloading => downloading,
        Section::Available => available,
    }
}

/// Focus stays on a section that lists models and otherwise moves on in the direction of travel, so the arrows skip
/// empty sections and startup does not land on an empty "Available" when everything is installed.
#[test]
fn focus_skips_empty_sections() {
    assert_eq!(Section::Available.first_with_models(Section::next, counts(25, 0, 0)), Section::Installed);
    assert_eq!(Section::Available.first_with_models(Section::next, counts(25, 0, 3)), Section::Available);
    assert_eq!(Section::Available.prev().first_with_models(Section::prev, counts(25, 0, 3)), Section::Installed);
    assert_eq!(Section::Installed.next().first_with_models(Section::next, counts(25, 0, 3)), Section::Available);
    assert_eq!(Section::Installed.next().first_with_models(Section::next, counts(25, 0, 0)), Section::Installed);
    assert_eq!(Section::Downloading.first_with_models(Section::next, counts(0, 0, 0)), Section::Downloading);
}
