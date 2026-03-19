use super::*;

#[test]
fn test_direct_swaps() {
    let sources = [0, 2, 4];
    let destinations = [1, 3, 5];
    let swaps = create_swaps_direct(&sources, &destinations);
    assert_eq!(swaps.len(), 3);
    assert_eq!(swaps[0].source, 0);
    assert_eq!(swaps[0].destination, 1);
    assert_eq!(swaps[1].source, 2);
    assert_eq!(swaps[1].destination, 3);
    assert_eq!(swaps[2].source, 4);
    assert_eq!(swaps[2].destination, 5);
}
