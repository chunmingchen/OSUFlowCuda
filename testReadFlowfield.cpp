//
//  testReadFlowfield.cpp - interpolation validation
//
//  Created by Jimmy Chen on 12/22/11.
//


#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>
#include "OSUFlow.h"
#include "OSUFlowCuda.h"
#include "cp_time.h"

#include <list>
#include <iterator>

using namespace std;

int main(int argc, char**argv) {
    srand(Timer::getTimeMS());
    
    VECTOR3 minLen, maxLen; 
    
    // == Init OSUFlow ==
    printf("hello! entering testmain...\n"); 
    
    OSUFlow *osuflow = new OSUFlow(); 
    
    printf("read file %s\n", argv[1]); 
    osuflow->LoadData((const char*)argv[1], true); //true: a steady flow field 
    osuflow->Boundary(minLen, maxLen); 
    osuflow->SetIntegrationParams(1,1); // fixed step size
    printf(" volume boundary X: [%f %f] Y: [%f %f] Z: [%f %f]\n", 
           minLen[0], maxLen[0], minLen[1], maxLen[1], 
           minLen[2], maxLen[2]); 
    float from[3], to[3]; 
    from[0] = minLen[0];   from[1] = minLen[1];   from[2] = minLen[2]; 
    to[0] = maxLen[0];   to[1] = maxLen[1];   to[2] = maxLen[2]; 

    // == Init OSUFlowCuda ==
    OSUFlowCuda *osuflowCuda = new OSUFlowCuda();
    
    printf("read file %s\n", argv[1]); 
    osuflowCuda->LoadData((const char*)argv[1], true); //true: a steady flow field 
    osuflowCuda->Boundary(minLen, maxLen); 
    osuflowCuda->SetIntegrationParams(.1,.1); // fixed step size
    printf(" volume boundary X: [%f %f] Y: [%f %f] Z: [%f %f]\n", 
           minLen[0], maxLen[0], minLen[1], maxLen[1], 
           minLen[2], maxLen[2]); 
    
    
    float step=.01f;
    float x,y,z;
    for (x=minLen[0]; x<=maxLen[0]; x+=step)
        for (y=minLen[1]; y<=maxLen[1]; y+=step)
            for (z=minLen[2]; z<=maxLen[2]; z+=step)
            {
                VECTOR3 vec1, vec2;
                int r1= osuflow->GetFlowField()->at_phys(VECTOR3(x,y,z), 0, vec1);
                int r2= osuflowCuda->at_phys(VECTOR3(x,y,z), 0, vec2);
                printf("Pos: %f %f %f, OSUFlow(r=%d): (%f %f %f), OSUFlowCuda(r=%d): (%f %f %f)\n", x,y,z, 
                       r1, vec1[0], vec1[1], vec1[2], r2, vec2[0], vec2[1], vec2[2]);
                if (!(vec1==vec2)) {
                    printf("**value mismatch****\n");
                    getchar();
                }
            }
    return 0;
}
