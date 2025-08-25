// Directed graph (each unordered pair of nodes is saved once): web-Stanford.txt 
// Stanford web graph row 2002
// Nodes: 281903 Edges: 2312497

#include "include/pagerank.h"
#include <vector>
#include <algorithm>

u_int64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * (u_int64_t)1000000 + tv.tv_usec;
}

extern void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* ptr, const int* row, const int* col, int gpus);
extern int MONTE_CARLO;

void csc2csr(const int nodes, const int edges, const int* rowdeg, int* rowptr, int* row, int* col) 
{

    int* new_row = new int[edges];
    int* new_col = new int[edges];
    int* index = new int[nodes];

    rowptr[0] = 0;
    // 类似前缀和
    for (int i = 0; i < nodes; i++) {
        rowptr[i + 1] = rowptr[i] + rowdeg[i];
        index[i] = 0;
    }

    for (int i = 0; i < edges; i++) {
        new_row[rowptr[row[i]] + index[row[i]]] = row[i];
        new_col[rowptr[row[i]] + index[row[i]]] = col[i];
        index[row[i]]++;
    }

    for (int i = 0; i < edges; i++) {
        row[i] = new_row[i];
        col[i] = new_col[i];
    }

    delete [] new_row;
    delete [] new_col;
    delete [] index;

    /*
    for (int n = 0; n < nodes; n++) {
        for (int i = rowptr[new_row[n]]; i < rowptr[new_row[n]]; i++) {
            int min = i;
            for (int j = i + 1; j < rowptr[new_row[n]]; j++) {
                if (new_col[min] > new_col[j]) {
                    min = j;
                }
            }
            std::swap(new_col[i], new_col[min]);
        }
    }
    */
}

void init_data(int nodes, int edges, char *filename, float* value, int* rowdeg, int* colptr, int* row , int* col)
{
    FILE* file = fopen(filename, "r");
    assert(file != NULL);

    for (int i = 0; i < nodes; i++) {
        rowdeg[i] = 0;
        value[i] = alpha;
    }

    fread(row, sizeof(int), edges, file);
    fread(col, sizeof(int), edges, file);

    int j = 0;
    colptr[0] = 0;

    for (int i = 0; i < edges; i++) {
        rowdeg[row[i]]++;
        while (j < col[i]) {
            colptr[++j] = i;
        }
    }

    colptr[nodes] = edges;
}

int main(int argc, char* argv[])
{   
    if (argc != 9 || strcasecmp("-f", argv[1]) || strcasecmp("-n", argv[3]) || strcasecmp("-e", argv[5]) || strcasecmp("-c", argv[7])) {
        fprintf(stderr, "Usage ./pagerank -f (file name) -n (number of nodes) -e (number of edges)\n");
        return 1;
    }

    char *filename = argv[2];
    // char filename[256];  // 预留足够空间
    // sprintf(filename, "/mnt/7T/tianyu/pagerank/18/%s", argv[2]);
    int nodes = atoi(argv[4]) + 1;
    int edges = atoi(argv[6]);
    int gpus = atoi(argv[8]); //rank数

    float* value = new float[nodes];
    int* rowdeg = new int[nodes];
    int* ptr = new int[nodes + 1];
    int* row = new int[edges];
    int* col = new int[edges];

    u_int64_t start_t, total_t;

    start_t = GetTimeStamp();
    init_data(nodes, edges, filename, value, rowdeg, ptr, row, col);
    total_t = GetTimeStamp() - start_t;
    printf("DataStruct: %s\nStart Warm Up GPU .... \n", (MONTE_CARLO) ? "CSR" : "CSC!");
    printf("I/O time usage: %.4f seconds\n", (float)total_t / 1e6);
    if (MONTE_CARLO) {
        csc2csr(nodes, edges, rowdeg, ptr, row, col);
    }


    start_t = GetTimeStamp();
    for(int i = 0; i < 10; i++) {
        pagerank(nodes, edges, value, rowdeg, ptr, row, col, gpus);
    }
    total_t = GetTimeStamp() - start_t;
    printf("WARMUP time usage: %.4f seconds\n", (float)total_t / 1e6);


    int epochs = 1;
    float average_time = 0, average_gteps = 0;
    for(int i = 0; i < epochs; i++)
    {
        start_t = GetTimeStamp();
        init_data(nodes, edges, filename, value, rowdeg, ptr, row, col);
        total_t = GetTimeStamp() - start_t;
        // printf("I/O         time usage: %.4f seconds\n", (float)total_t / 1e6);

        if (MONTE_CARLO) {
            csc2csr(nodes, edges, rowdeg, ptr, row, col);
        }
        int iterations = 10;
        start_t = GetTimeStamp();
        for(int i = 0; i < iterations; i++) {
            pagerank(nodes, edges, value, rowdeg, ptr, row, col, gpus);
        }
        total_t = GetTimeStamp() - start_t;
        float used_time = (float)total_t / 1e6; // trans microsecond to second
        float gteps = edges * iterations / used_time / 1e9;
        average_time += used_time, average_gteps += gteps;
        // printf("Cal time usage: %.4f seconds, GTEPS: %.4f\n", used_time, gteps);
    }
    average_gteps /= epochs, average_time /= epochs;
    printf("================================ DATAFLOW METHOD =======================================\n");
    printf("\t\tGraph nodes = %d, edges = %d, Env_Cards: %d\n", nodes-1, edges, gpus);
    printf("\t\tGTEPS: %.4f\n", average_gteps);

    


    // 创建一个包含节点 ID 和 PageRank 值的结构体数组
    std::vector<std::pair<int, float>> node_rank;
    for (int i = 0; i < nodes; ++i) {
        node_rank.push_back(std::make_pair(i, value[i]));
    }

    // 根据 PageRank 值对节点进行排序，从高到低
    // std::sort(node_rank.begin(), node_rank.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
    //     return a.second > b.second;  // 按照第二个元素（PageRank 值）降序排列
    // });

    // // 打印出排名前 20 的节点及其 PageRank 值
    // printf("Rank | Node ID | Score\n");
    // for(int i = 0; i < 30; i++) printf("-");
    // puts("");
    // for (int i = 0; i < 20; ++i) {
    //     printf("%4d | %7d | %.6f\n", i + 1, node_rank[i].first + 1, node_rank[i].second / 1e6);
    // }

    FILE* fout = fopen("rank_result.bin", "wb");
    if (!fout) {
        perror("Failed to open output file");
        exit(1);
    }
    fwrite(value, sizeof(float), nodes, fout);
    fclose(fout);

    delete [] value;
    delete [] rowdeg;
    delete [] row;
    delete [] col;

    return 0;
}