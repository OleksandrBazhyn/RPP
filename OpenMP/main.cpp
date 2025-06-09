#include <omp.h>
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

// Read all lines from a file into a vector
std::vector<std::string> read_lines(const std::string& fname) {
    std::ifstream fin(fname);
    std::vector<std::string> lines;
    std::string buf;
    while (std::getline(fin, buf))
        lines.push_back(buf);
    return lines;
}

int main() {
    printf("OpenMP\n");

    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::time_point<clock> time_point;

    time_point startTP = clock::now();

    // 1. Read both files into memory
    std::vector<std::string> lines1 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\a.txt");
    std::vector<std::string> lines2 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\b.txt");
    int n1 = lines1.size();
    int n2 = lines2.size();

    // 2. Prepare a multimap of hashes for all non-empty lines in b.txt for fast lookup
    std::unordered_multimap<uint64_t, int> hash2idx;
    for (int i = 0; i < n2; ++i)
        if (!lines2[i].empty())
            hash2idx.emplace(string_hash(lines2[i]), i);

    // 3. Arrays for matching results
    std::vector<int> matches;       // indices of matching lines from lines1
    std::vector<bool> matched(n1, false); // mark which lines from lines1 have a match

    int non_empty_n1 = 0; // Number of non-empty lines in lines1

    // 4. Parallel loop: each thread compares a subset of non-empty lines1 to all of lines2 using OpenMP
#pragma omp parallel for schedule(dynamic) reduction(+:non_empty_n1)
    for (int i = 0; i < n1; ++i) {
        const std::string& s = lines1[i];
        if (s.empty()) continue; // Ignore empty lines in a.txt
        ++non_empty_n1;

        uint64_t h = string_hash(s);
        auto range = hash2idx.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            if (s == lines2[it->second]) {
                matched[i] = true; // found an exact match
                break;
            }
        }
    }

    // 5. Collect all matching non-empty indices
    for (int i = 0; i < n1; ++i)
        if (matched[i] && !lines1[i].empty())
            matches.push_back(i);

    // 6. Output the results
    double percent = non_empty_n1 ? (100.0 * matches.size() / non_empty_n1) : 0.0;
    std::cout << "Number of matching (non-empty) lines: " << matches.size()
        << " out of " << non_empty_n1 << " (" << percent << "%)" << std::endl;
    std::cout << "Matching non-empty lines in a.txt:" << std::endl << std::endl;
    std::unordered_set<int> was;
    for (int idx : matches) {
        if (was.insert(idx).second)
            std::cout << "  [" << idx + 1 << "] " << lines1[idx] << std::endl;
    }

    time_point endTP = clock::now();
    double elapsed_sec = std::chrono::duration<double>(endTP - startTP).count();
    std::cout << std::endl << "Elapsed time: " << elapsed_sec << " seconds" << std::endl;

    return 0;
}
