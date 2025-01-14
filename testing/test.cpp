#include <cstdint>
#include "ozblas.h"
#include <random>
#include <iostream>
#include <quadmath.h>
#include <omp.h>

int main(int argc, char** argv)
{
    omp_set_num_threads(1);

    //uint64_t ozblasworkmemory = 2e9; // 2e9 and 0.25e9 give accurate results but 1e9 and 0.5e9 do not
    uint64_t ozblasworkmemory = 0.5e9; // 2e9 and 0.25e9 give accurate results but 1e9 and 0.5e9 do not

    // Standard mersenne twister engine seeded with default seed (results should be reproducible):
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<__float128> binary128numbers {3.1415926535897932384626433832795028q, 1.0q / 3.0q, powq(2, -16382)};
    int m = 100;//100;
    //int k = 100;
    int k = 25000;
    std::vector<__float128> A(m * k);
    std::vector<__float128> B(k * m);
    __float128* Avals = A.data();
    __float128* Bvals = B.data();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            Avals[i*k+j] = (powq(10.0q, -10 + (int)(20 * dis(gen)))*dis(gen) + binary128numbers[(int)(3 * dis(gen))]);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < m; ++j)
            Bvals[i*m+j] = (powq(10.0q, -10 + (int)(20 * dis(gen)))*dis(gen) + binary128numbers[(int)(3 * dis(gen))]);

    std::cout << "A and B initialised" << std::endl;

    // Compute C = A * B using a simple triple-nested loop
    std::vector<__float128> C(m * m, 0.0q);
    __float128* Cvals = C.data();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            for (int l = 0; l < k; ++l)
                Cvals[i*m+j] += Avals[i*k+l] * Bvals[l*m+j];

    std::cout << "C computed using triple-nested loop" << std::endl;

    // Compute C = A * B using OzBLAS
    std::vector<__float128> Cozblas(m * m, 0.0q);
    __float128* Cozblasvals = Cozblas.data();
    ozblasHandle_t options;
    ozblasCreate(&options, ozblasworkmemory);
    options.splitModeFlag = 0;//3;
    options.fastModeFlag = 0;
    options.sumModeFlag = 0;//3;
    options.nSplitMax = 100;//100;
    ozblasRgemm<__float128,double>(&options, 'N', 'N', m, m, k, 1.0, Bvals, m, Avals, k, 0.0, Cozblasvals, m);

	printf ("nSplitA = %f\n", options.nSplitA);
	printf ("nSplitB = %f\n", options.nSplitB);
	printf ("nSplitC = %f\n", options.nSplitC);
	printf ("n_comp = %f\n", options.n_comp);
	printf ("mbk = %d\n", options.mbk);
	printf ("nbk = %d\n", options.nbk);

    ozblasDestroy(&options);

    std::cout << "C computed using OzBLAS (memory available: " << ozblasworkmemory/1.0e9 << " GB)" << std::endl;

    // Check if the results are the same
    __float128 maxreldiff = 0.0q;
    __float128 maxdiff = 0.0q;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            __float128& ozblasval = Cozblasvals[i*m+j];
            __float128& tripleloopval = Cvals[i*m+j];
            if (fabsq(tripleloopval) > 1e-6)
            {
                __float128 reldiff = fabsq(tripleloopval - ozblasval) / fabsq(tripleloopval);
                if (reldiff > maxreldiff)
                    maxreldiff = reldiff;
            }
            else
            {
                __float128 diff = fabsq(tripleloopval - ozblasval);
                if (diff > maxdiff)
                    maxdiff = diff;
            }
        }
    }

    std::cout << "maxreldiff: " << (double)maxreldiff << " maxdiff: " << (double)maxdiff << std::endl;
}
