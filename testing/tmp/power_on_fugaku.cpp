#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include "pwr.h"

int32_t
main (int32_t argc, char **argv)
{
	// only on Fugaku
	PWR_Cntxt cntxt = NULL;
	PWR_Obj obj = NULL;
	double energy0 = 0.;
	double energy1 = 0.;
	double avg_power = 0.;
	PWR_Time ts0 = 0;
	PWR_Time ts1 = 0;

	PWR_CntxtInit(PWR_CNTXT_DEFAULT, PWR_ROLE_APP, "app", &cntxt); // initialization
	PWR_CntxtGetObjByName(cntxt, "plat.node", &obj); // get object

			PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy0, &ts0); // start
			sleep(10);

			PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy1, &ts1); // end
			avg_power = (energy1 - energy0) / ((ts1 - ts0) * 1e-9);
			printf ("(avgPwr %1.3e)\n", avg_power);

	PWR_CntxtDestroy(cntxt);

	return 0;
}

