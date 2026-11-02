import Foundation

enum StandardStreams {
    static func run(
        input: (FileHandle) -> Void, output: @escaping (UInt8, Data) -> Void, _ body: () -> Int32
    ) -> Int32 {
        let saved = (0...2).map { dup(Int32($0)) }
        let pipes = (0...2).map { _ in Pipe() }
        dup2(pipes[0].fileHandleForReading.fileDescriptor, 0)
        dup2(pipes[1].fileHandleForWriting.fileDescriptor, 1)
        dup2(pipes[2].fileHandleForWriting.fileDescriptor, 2)
        try? pipes[0].fileHandleForReading.close()
        try? pipes[1].fileHandleForWriting.close()
        try? pipes[2].fileHandleForWriting.close()
        clearerr(stdin)

        input(pipes[0].fileHandleForWriting)
        let readers = DispatchGroup()
        for stream in 1...2 {
            DispatchQueue.global().async(group: readers) {
                let handle = pipes[stream].fileHandleForReading
                var data = handle.availableData
                while !data.isEmpty {
                    output(UInt8(stream), data)
                    data = handle.availableData
                }
            }
        }

        let status = body()
        fflush(stdout)
        for stream in 0...2 {
            dup2(saved[stream], Int32(stream))
        }
        readers.wait()
        for descriptor in saved {
            close(descriptor)
        }
        return status
    }
}
