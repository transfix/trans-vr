
#include <SweetLBIE/octree.h>

void sweetLBIE::generateOctree(sweetMesh::hexMesh& octreeMesh, double step, VolMagick::Volume& vol, double isoval) {
    octreeMesh.clear();
    for (double x=vol.XMin(); x<vol.XMax(); x+=step) {
        for (double y=vol.YMin(); y<vol.YMax(); y+=step) {
            for (double z=vol.ZMin(); z<vol.ZMax(); z+=step) {
                try {
                    sweetMesh::hexVertex::hexVertex v0(x, y, z);
                    sweetMesh::hexVertex::hexVertex v1(x+step, y, z);
                    sweetMesh::hexVertex::hexVertex v2(x+step, y+step, z);
                    sweetMesh::hexVertex::hexVertex v3(x, y+step, z);
                    sweetMesh::hexVertex::hexVertex v4(x, y, z+step);
                    sweetMesh::hexVertex::hexVertex v5(x+step, y, z+step);
                    sweetMesh::hexVertex::hexVertex v6(x+step, y+step, z+step);
                    sweetMesh::hexVertex::hexVertex v7(x, y+step, z+step);
                    octreeMesh.addHex(v0, v1, v2, v3, v4, v5, v6, v7);
                } catch (VolMagick::IndexOutOfBounds& e) {}
            }
        }
    }
}

bool sweetLBIE::testOctreeHex(double x, double y, double z, double step, double isoval, VolMagick::Volume& vol) {
    bool middleSign, s0, s1, s2, s3, s4, s5, s6, s7;

    s0 = (vol.interpolate(x,y,z)		> isoval);
    s1 = (vol.interpolate(x, y+step, z)		> isoval);
    s2 = (vol.interpolate(x+step,y+step,z)	> isoval);
    s3 = (vol.interpolate(x,y+step,z)		> isoval);
    s4 = (vol.interpolate(x,y,z+step)		> isoval);
    s5 = (vol.interpolate(x+step,y,z+step)	> isoval);
    s6 = (vol.interpolate(x+step,y+step,z+step)	> isoval);
    s7 = (vol.interpolate(x,y+step,z+step)	> isoval);

    //test cube
    middleSign = ( vol.interpolate(x+step/2.0, y+step/2.0, z+step/2.0) > isoval );
    if ( !((((((((middleSign==s0) || middleSign==s1) || middleSign==s2) || middleSign==s3) || middleSign==s4) || middleSign==s5) || middleSign==s6) || middleSign==s7) ) {
        std::cout << "failed at cube test\n";
        return false;
    }
    //test faces
    middleSign = ( vol.interpolate(x+step/2.0, y+step/2.0, z)	> isoval);
    if ( ! (middleSign==s0 || (middleSign==s1 || (middleSign==s2 || (middleSign==s3)))) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y, z+step/2.0)	> isoval);
    if ( ! (middleSign==s0 || (middleSign==s4 || (middleSign==s5 || (middleSign==s1)))) )
        return false;
    middleSign = ( vol.interpolate(x+step, y+step/2.0, z+step/2.0)	> isoval);
    if ( ! (middleSign==s1 || (middleSign==s5 || (middleSign==s6 || (middleSign==s2)))) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y+step, z+step/2.0)	> isoval);
    if ( ! (middleSign==s2 || (middleSign==s6 || (middleSign==s7 || (middleSign==s3)))) )
        return false;
    middleSign = ( vol.interpolate(x, y+step/2.0, z+step/2.0)	> isoval);
    if ( ! (middleSign==s0 || (middleSign==s3 || (middleSign==s7 || (middleSign==s4)))) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y+step/2.0, z+step)	> isoval);
    if ( ! (middleSign==s4 || (middleSign==s7 || (middleSign==s6 || (middleSign==s5)))) )
        return false;
    //test edges
    middleSign = ( vol.interpolate(x+step/2.0, y, z) > isoval);
    if ( ! (middleSign==s0 || middleSign==s1) )
        return false;
    middleSign = ( vol.interpolate(x+step, y+step/2.0, z) > isoval);
    if ( ! (middleSign==s1 || middleSign==s2) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y+step/2.0, z) > isoval);
    if ( ! (middleSign==s2 || middleSign==s3) )
        return false;
    middleSign = ( vol.interpolate(x, y+step/2.0, z) > isoval);
    if ( ! (middleSign==s3 || middleSign==s0) )
        return false;
    middleSign = ( vol.interpolate(x, y, z+step/2.0) > isoval);
    if ( ! (middleSign==s0 || middleSign==s4) )
        return false;
    middleSign = ( vol.interpolate(x+step, y, z+step/2.0) > isoval);
    if ( ! (middleSign==s1 || middleSign==s5) )
        return false;
    middleSign = ( vol.interpolate(x+step, y+step, z+step/2.0) > isoval);
    if ( ! (middleSign==s2 || middleSign==s6) )
        return false;
    middleSign = ( vol.interpolate(x, y+step, z+step/2.0) > isoval);
    if ( ! (middleSign==s3 || middleSign==s7) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y, z+step) > isoval);
    if ( ! (middleSign==s4 || middleSign==s5) )
        return false;
    middleSign = ( vol.interpolate(x+step, y+step/2.0, z+step) > isoval);
    if ( ! (middleSign==s5 || middleSign==s6) )
        return false;
    middleSign = ( vol.interpolate(x+step/2.0, y+step, z+step) > isoval);
    if ( ! (middleSign==s6 || middleSign==s7) )
        return false;
    middleSign = ( vol.interpolate(x, y+step/2.0, z+step) > isoval);
    if ( ! (middleSign==s7 || middleSign==s4) )
        return false;
    return true;
}

bool sweetLBIE::testOctreeStep(double step, double isoval, VolMagick::Volume& vol) {
    for (double x=vol.XMin(); x<vol.XMax(); x+=step) {
        for (double y=vol.YMin(); y<vol.YMax(); y+=step) {
            for (double z=vol.ZMin(); z<vol.ZMax(); z+=step) {
                try {
                    if ( ! testOctreeHex(x, y, z, step, isoval, vol) ) {
                        return false;
                    }
                } catch (VolMagick::IndexOutOfBounds& e) {}
            }
        }
    }
return true;
}

double sweetLBIE::computeStepSize(VolMagick::Volume& vol, double isoval) {
    double maxExtent, minStep, step;

    maxExtent = std::max( std::max(vol.XMax()-vol.XMin(), vol.YMax()-vol.YMin()), vol.ZMax()-vol.ZMin());
    minStep = std::min( std::min(vol.XSpan(), vol.YSpan()), vol.ZSpan());

    for (step=maxExtent; step>minStep; step/=2.0) {
        if ( ! testOctreeStep(step, isoval, vol) ) {
            return step*2.0;
        }
    }
    return minStep;
}

double sweetLBIE::getOctree(VolMagick::Volume& vol, sweetMesh::hexMesh& octreeMesh, double isoval) {
    double stepSize;

    stepSize = computeStepSize(vol, isoval);
    generateOctree(octreeMesh, stepSize, vol, isoval);
    return stepSize;
}
