implementing definitions;

public interface IGroupShared<T> {
    static func load(uint idx) -> T;
    static func store(uint idx, T value);
}
