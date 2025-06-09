#include <omp.h>
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
    // 1. Read both files into memory
    std::vector<std::string> lines1 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\a.txt");
    std::vector<std::string> lines2 = read_lines("C:\\My Source Code\\Навчання\\Університет\\РПП\\lab\\RPP\\b.txt");
    int n1 = lines1.size();
    int n2 = lines2.size();

    // 2. Prepare a multimap of hashes for all lines in b.txt for fast lookup
    std::unordered_multimap<uint64_t, int> hash2idx;
    for (int i = 0; i < n2; ++i)
        hash2idx.emplace(string_hash(lines2[i]), i);

    // 3. Arrays for matching results
    std::vector<int> matches;       // indices of matching lines from lines1
    std::vector<bool> matched(n1, false); // mark which lines from lines1 have a match

    // 4. Parallel loop: each thread compares a subset of lines1 to all of lines2 using OpenMP
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n1; ++i) {
        const std::string& s = lines1[i];
        uint64_t h = string_hash(s);
        auto range = hash2idx.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            if (s == lines2[it->second]) {
                matched[i] = true; // found an exact match
                break;
            }
        }
    }

    // 5. Collect all matching indices
    for (int i = 0; i < n1; ++i)
        if (matched[i]) matches.push_back(i);

    // 6. Output the results
    double percent = n1 ? (100.0 * matches.size() / n1) : 0.0;
    std::cout << "Number of matching lines: " << matches.size() << " out of " << n1
        << " (" << percent << "%)" << std::endl;
    std::cout << "Matching lines in a.txt:" << std::endl;
    std::unordered_set<int> was;
    for (int idx : matches) {
        if (was.insert(idx).second) // print each unique line only once
            std::cout << "  [" << idx + 1 << "] " << lines1[idx] << std::endl;
    }

    return 0;
}
