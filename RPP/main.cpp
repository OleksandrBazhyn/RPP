#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// --------- HASH PARAMETERS ---------
const uint64_t P = 31;                   // base (prime, for rolling hash)
const uint64_t MOD = 1000000007;         // large prime modulus

// --------- HASH FUNCTION (ROLLING) ---------
uint64_t poly_hash(const std::string& s) {
    uint64_t hash = 0, p_pow = 1;
    for (char c : s) {
        hash = (hash + (c - 'a' + 1) * p_pow) % MOD;
        p_pow = (p_pow * P) % MOD;
    }
    return hash;
}

// --------- MAIN MPI PROGRAM ---------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string text;
    int text_len = 0;
    std::vector<char> chunk;
    int chunk_size = 0;

    // Rank 0 reads the file/text
    if (world_rank == 0) {
        std::ifstream fin("big_text.txt");
        if (!fin) {
            std::cerr << "Can't open file!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::getline(fin, text, '\0');
        fin.close();
        text_len = (int)text.size();
    }

    // Broadcast text size to all ranks
    MPI_Bcast(&text_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate chunk sizes for each rank
    chunk_size = text_len / world_size + (world_rank < text_len % world_size ? 1 : 0);
    chunk.resize(chunk_size);

    // Scatterv setup
    std::vector<int> sendcounts, displs;
    if (world_rank == 0) {
        sendcounts.resize(world_size);
        displs.resize(world_size);
        int sum = 0;
        for (int i = 0; i < world_size; ++i) {
            sendcounts[i] = text_len / world_size + (i < text_len % world_size ? 1 : 0);
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Scatter the text data to all ranks
    MPI_Scatterv(text.data(),
        sendcounts.data(),
        displs.data(),
        MPI_CHAR,
        chunk.data(),
        chunk_size,
        MPI_CHAR,
        0,
        MPI_COMM_WORLD);

    // Each rank computes the hash of its chunk
    std::string chunk_str(chunk.begin(), chunk.end());
    uint64_t local_hash = poly_hash(chunk_str);

    // Gather all hashes to rank 0
    std::vector<uint64_t> all_hashes;
    if (world_rank == 0)
        all_hashes.resize(world_size);

    MPI_Gather(&local_hash, 1, MPI_UINT64_T, all_hashes.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Rank 0 can now compare or combine the hashes
    if (world_rank == 0) {
        std::cout << "Chunk hashes: ";
        for (auto h : all_hashes) std::cout << h << " ";
        std::cout << std::endl;
        // Optionally, combine hashes for a total file hash
        // (if you want a "global" hash, you must use proper rolling hash combining logic)
    }

    MPI_Finalize();
    return 0;
}
