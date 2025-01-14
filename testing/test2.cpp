#include <cstdint>
#include "ozblas.h"
#include <random>
#include <iostream>
#include <quadmath.h>
#include <omp.h>

int main(int argc, char** argv)
{
    omp_set_num_threads(1);
   
    uint64_t ozblasworkmemory = 1e9;
   
    // Standard mersenne twister engine seeded with default seed (results should be reproducible):
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<__float128> binary128numbers {3.1415926535897932384626433832795028q, 1.0q / 3.0q, powq(2, -16382)};
    size_t m = 1;//1;
    size_t k = 3;//3;
    size_t n = 464010;//464010;
    std::vector<__float128> A(m * k);
    std::vector<__float128> B(k * n);
    __float128* Avals = A.data();
    __float128* Bvals = B.data();
    Avals[0] = 100.00000000000001110223024625156663683148108873915q;
    Avals[1] = -1.4920616989674252377920030343259011872014294491837e-15q;
    Avals[2] = 3.4483369642628911869358020310414255559771235739917e-25q;
    //for (int i = 0; i < k; ++i)
    //    for (int j = 0; j < n; ++j)
    for (size_t i = 0; i < k; ++i){
        for (size_t j = 0; j < n; ++j){
            Bvals[i*n+j] = (powq(10.0q, -10 + (int)(20 * dis(gen)))*dis(gen) + binary128numbers[(int)(3 * dis(gen))]);
//            char str[128];
//           if(isnanq(Bvals[i*n+j]))quadmath_snprintf(str,128,"%.3e",Bvals[i*n+j]);
            }
            }
           
    std::cout << "A and B initialised" << std::endl;
   
    // Compute C = A * B using a simple triple-nested loop
    std::vector<__float128> C(m * n, 0.0q);
    __float128* Cvals = C.data();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                Cvals[i*n+j] += Avals[i*k+l] * Bvals[l*n+j];
               
    std::cout << "C computed using triple-nested loop" << std::endl;
   
    // Compute C = A * B using OzBLAS
    std::vector<__float128> Cozblas(m * n, 0.0q);
    __float128* Cozblasvals = Cozblas.data();
    ozblasHandle_t options;
    ozblasCreate(&options, ozblasworkmemory);
//    options.sumModeFlag = 0;
//	options.splitModeFlag = 3;
    options.nSplitMax = 50;//100;
    ozblasRgemm<__float128,double>(&options, 'N', 'N', n, m, k, 1.0, Bvals, n, Avals, k, 0.0, Cozblasvals, n);
    printf ("nSplitA=%f nSplitB=%f mbk=%d nbk=%d\n", options.nSplitA, options.nSplitB, options.mbk, options.nbk);
    ozblasDestroy(&options);
   
    std::cout << "C computed using OzBLAS (memory available: " << ozblasworkmemory/1.0e9 << " GB)" << std::endl;

    // Check if the results are the same
    __float128 maxreldiff = 0.0q;
    __float128 maxdiff = 0.0q;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            __float128& ozblasval = Cozblasvals[i*n+j];
            __float128& tripleloopval = Cvals[i*n+j];
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
