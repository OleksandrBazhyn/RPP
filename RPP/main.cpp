#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

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

    // --- 2. Passes the row count to all processes
    MPI_Bcast(&n1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- 3. Divides work: each process gets a subrange of indices
    int chunk = (n1 + nprocs - 1) / nprocs;
    int begin = rank * chunk;
    int end = std::min(begin + chunk, n1);

    // --- 4. Passes its lines to each process (scatter-like, via Bcast)
    std::vector<std::string> my_lines1(chunk);
    if (rank == 0) {
        for (int r = 1; r < nprocs; ++r) {
            int b = r * chunk;
            int e = std::min(b + chunk, n1);
            int cnt = e - b;
            for (int i = 0; i < cnt; ++i) {
                int len = lines1[b + i].size();
                MPI_Send(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(lines1[b + i].c_str(), len, MPI_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
        // Copy own chunk
        for (int i = begin; i < end; ++i) my_lines1[i - begin] = lines1[i];
    }
    else {
        for (int i = begin; i < end; ++i) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string buf(len, ' ');
            MPI_Recv(&buf[0], len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            my_lines1[i - begin] = buf;
        }
    }

    // --- 5. Rank 0 transmits all lines of the second file (broadcast, because everyone compares with everyone)
    int* line2_lens = new int[n2];
    if (rank == 0) for (int i = 0; i < n2; ++i) line2_lens[i] = lines2[i].size();
    MPI_Bcast(line2_lens, n2, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::string> all_lines2(n2);
    for (int i = 0; i < n2; ++i) {
        if (rank == 0)
            all_lines2[i] = lines2[i];
        else {
            all_lines2[i].resize(line2_lens[i]);
        }
        MPI_Bcast(&all_lines2[i][0], line2_lens[i], MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    delete[] line2_lens;

    // --- 6. We form a set of hashes for the second file for quick search
    std::unordered_multimap<uint64_t, int> hash2idx;
    for (int i = 0; i < n2; ++i)
        hash2idx.emplace(string_hash(all_lines2[i]), i);

    // --- 7. Compare hashes (and text for confirmation) for our strings
    std::vector<int> local_matches; // matching line numbers from lines1 (local)

    for (int i = 0; i < end - begin; ++i) {
        const std::string& s = my_lines1[i];
        uint64_t h = string_hash(s);
        auto range = hash2idx.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            if (s == all_lines2[it->second]) {
                local_matches.push_back(begin + i); // we save the global number
                break;
            }
        }
    }

    // --- 8. We pass the found matches to rank 0 (gather)
    // We pass the quantity, and then the numbers themselves
    int local_cnt = local_matches.size();
    std::vector<int> all_counts(nprocs);
    MPI_Gather(&local_cnt, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(nprocs), recvbuf;
    int total = 0;
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
        double percent = n1 ? (100.0 * total / n1) : 0.0;
        std::cout << "Matching rows: " << total << " from " << n1
            << " (" << percent << "%)" << std::endl;
        std::cout << "The following lines match a.txt:" << std::endl;
        std::unordered_set<int> was;
        for (int idx : recvbuf) {
            if (was.insert(idx).second) // unique numbers
                std::cout << "  [" << idx + 1 << "] " << lines1[idx] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
