////////////////////////////////////////////////////////////////////////////
//
// Compare OSUFlowCuda and OSUFlow computation time
// Chun-Ming Chen
//
///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h> 
#include "OSUFlow.h"
#include "OSUFlowCuda.h"
#include "cp_time.h"

#include <list>
#include <iterator>

int computeStreamlines(OSUFlow *osuflow)
{
    VECTOR3 minLen, maxLen;

  osuflow->Boundary(minLen, maxLen);
  //printf(" volume boundary X: [%f %f] Y: [%f %f] Z: [%f %f]\n",
  //                              minLen[0], maxLen[0], minLen[1], maxLen[1],
  //                              minLen[2], maxLen[2]);
  float from[3], to[3];
  from[0] = minLen[0];   from[1] = minLen[1];   from[2] = minLen[2];
  to[0] = maxLen[0];   to[1] = maxLen[1];   to[2] = maxLen[2];
  printf("Generating 100000 seeds\n");
  osuflow->SetRandomSeedPoints(from, to, 100000);
  int nSeeds;
  VECTOR3* seeds = osuflow->GetSeeds(nSeeds);

  list<vtListSeedTrace*> list;
  osuflow->SetIntegrationParams(1, 5);
  Timer timer;
  timer.start();
  osuflow->GenStreamLines(list , FORWARD_DIR, 50, 0);
  timer.end();
  printf(" done integrations\n");
  printf("Time: %lf secs\n", timer.getElapsedUS()/1000000.0);

}


int computeStreamlinesCuda(OSUFlowCuda *osuflow)
{
  VECTOR3 minLen, maxLen;

  osuflow->Boundary(minLen, maxLen);
  //printf(" volume boundary X: [%f %f] Y: [%f %f] Z: [%f %f]\n",
  //                              minLen[0], maxLen[0], minLen[1], maxLen[1],
  //                              minLen[2], maxLen[2]);
  float from[3], to[3];
  from[0] = minLen[0];   from[1] = minLen[1];   from[2] = minLen[2];
  to[0] = maxLen[0];   to[1] = maxLen[1];   to[2] = maxLen[2];
  printf("Generating 100000 seeds\n");
  osuflow->SetRandomSeedPoints(from, to, 100000);
  int nSeeds;
  VECTOR3* seeds = osuflow->GetSeeds(nSeeds);

  list<vtListSeedTrace*> list;
  osuflow->SetIntegrationParams(1, 5);
  Timer timer;
  timer.start();
  osuflow->GenStreamLines(list, FORWARD_DIR, 50, 0);

  timer.end();
  printf(" done integrations\n");
  printf("Time: %lf secs\n", timer.getElapsedUS()/1000000.0);

}

int	// ADD-BY-LEETEN 12/20/2011
main(int argc, char**argv) {


  OSUFlow *osuflow = new OSUFlow();
  OSUFlowCuda *osuflowcuda = new OSUFlowCuda();
  printf("read file %s\n", argv[1]);
  osuflow->LoadData((const char*)argv[1], true); //true: a steady flow field
  osuflowcuda->LoadData((const char*)argv[1], true); //true: a steady flow field

  printf("Running gpu osuflow...\n")  ;
  computeStreamlinesCuda(osuflowcuda);
  printf("Running cpu osuflow...\n")  ;
  computeStreamlines(osuflow);


  return 0;	// ADD-BY-LEETEN 12/20/2011
}
