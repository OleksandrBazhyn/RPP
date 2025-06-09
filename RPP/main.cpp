#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>

// Simple polynomial rolling hash for a string (like Rabin-Karp)
uint64_t string_hash(const std::string& s) {
    const uint64_t P = 31, MOD = 1e9 + 9;
    uint64_t hash = 0, p_pow = 1;
    for (char c : s) {
        hash = (hash + (uint64_t)(c - 'a' + 1) * p_pow) % MOD;
        p_pow = (p_pow * P) % MOD;
    }
    return hash;
}

// Read all lines from a file
std::vector<std::string> read_lines(const std::string& fname) {
    std::ifstream fin(fname);
    std::vector<std::string> lines;
    std::string buf;
    while (std::getline(fin, buf))
        lines.push_back(buf);
    return lines;
}

int main(int argc, char** argv) {
    printf("MPI\n");

    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::time_point<clock> time_point;

    time_point startTP = clock::now();

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // --- 1. Rank 0 reads both files and splits lines
    std::vector<std::string> lines1, lines2;
    int n1 = 0, n2 = 0;
    if (rank == 0) {
        lines1 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\a.txt");
        lines2 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\b.txt");
        n1 = lines1.size();
        n2 = lines2.size();
    }

    // --- 2. Pass row count to all processes
    MPI_Bcast(&n1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- 3. Divide work: each process gets a subrange of indices
    int chunk = (n1 + nprocs - 1) / nprocs;
    int begin = rank * chunk;
    int end = std::min(begin + chunk, n1);

    // --- 4. Scatter-like lines transmission
    std::vector<std::string> my_lines1(chunk);
    if (rank == 0) {
        for (int r = 1; r < nprocs; ++r) {
            int b = r * chunk;
            int e = std::min(b + chunk, n1);
            int cnt = e - b;
            for (int i = 0; i < cnt; ++i) {
                int len = lines1[b + i].size();
                MPI_Send(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                if (len)
                    MPI_Send(lines1[b + i].c_str(), len, MPI_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
        for (int i = begin; i < end; ++i)
            my_lines1[i - begin] = lines1[i];
    }
    else {
        for (int i = begin; i < end; ++i) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string buf(len, ' ');
            if (len)
                MPI_Recv(&buf[0], len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            my_lines1[i - begin] = buf;
        }
    }

    // --- 5. Rank 0 broadcasts all lines of the second file (for matching)
    int* line2_lens = new int[n2];
    if (rank == 0) for (int i = 0; i < n2; ++i) line2_lens[i] = lines2[i].size();
    MPI_Bcast(line2_lens, n2, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::string> all_lines2(n2);
    for (int i = 0; i < n2; ++i) {
        if (rank == 0)
            all_lines2[i] = lines2[i];
        else
            all_lines2[i].resize(line2_lens[i]);
        if (line2_lens[i])
            MPI_Bcast(&all_lines2[i][0], line2_lens[i], MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    delete[] line2_lens;

    // --- 6. Build set of hashes for all non-empty lines in b.txt
    std::unordered_multimap<uint64_t, int> hash2idx;
    for (int i = 0; i < n2; ++i)
        if (!all_lines2[i].empty())
            hash2idx.emplace(string_hash(all_lines2[i]), i);

    // --- 7. Compare only non-empty lines from my_lines1
    std::vector<int> local_matches; // matching line numbers from lines1 (local)
    int local_nonempty = 0;
    for (int i = 0; i < end - begin; ++i) {
        const std::string& s = my_lines1[i];
        if (s.empty()) continue; // skip empty lines
        ++local_nonempty;

        uint64_t h = string_hash(s);
        auto range = hash2idx.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            if (s == all_lines2[it->second]) {
                local_matches.push_back(begin + i); // save global index
                break;
            }
        }
    }

    // --- 8. Gather match counts and local_nonempty for percent calculation
    int local_cnt = local_matches.size();
    int all_nonempty = 0, total = 0;
    MPI_Reduce(&local_nonempty, &all_nonempty, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<int> all_counts(nprocs);
    MPI_Gather(&local_cnt, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(nprocs), recvbuf;
    if (rank == 0) {
        for (int i = 0; i < nprocs; ++i) {
            displs[i] = total;
            total += all_counts[i];
        }
        recvbuf.resize(total);
    }
    MPI_Gatherv(local_matches.data(), local_cnt, MPI_INT,
        recvbuf.data(), all_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // --- 9. Counting and outputting the result to rank 0
    if (rank == 0) {
        double percent = all_nonempty ? (100.0 * total / all_nonempty) : 0.0;
        std::cout << "Matching (non-empty) lines: " << total << " from " << all_nonempty
            << " (" << percent << "%)" << std::endl;
        std::cout << "Matching non-empty lines in a.txt:" << std::endl << std::endl;
        std::unordered_set<int> was;
        for (int idx : recvbuf) {
            if (was.insert(idx).second)
                std::cout << "  [" << idx + 1 << "] " << lines1[idx] << std::endl;
        }
    }

    MPI_Finalize();

    time_point endTP = clock::now();
    if (rank == 0) {
        double elapsed_sec = std::chrono::duration<double>(endTP - startTP).count();
        std::cout << std::endl << "Elapsed time: " << elapsed_sec << " seconds" << std::endl;
    }

    return 0;
}
