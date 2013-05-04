#!/bin/sh
export SCRIPT_LOCATION=.
find . -name "*.h" -exec perl -i $SCRIPT_LOCATION/header-fix.pl \{\} \; -print
find ../src -name "*.cpp" -exec perl -i $SCRIPT_LOCATION/header-fix.pl \{\} \; -print
find ../src -name "*.c" -exec perl -i $SCRIPT_LOCATION/header-fix.pl \{\} \; -print

#Now that most are converted already, we have a few exceptions!
svn revert -R FastContouring
svn revert -R ../src/FastContouring
svn revert -R LBIE 
svn revert -R ../src/LBIE
svn revert -R VolumeFileTypes 
svn revert -R ../src/VolumeFileTypes
svn revert -R Curation
svn revert -R ../src/Curation
svn revert -R PocketTunnel
svn revert -R ../src/PocketTunnel
svn revert -R Skeletonization
svn revert -R ../src/Skeletonization
svn revert -R TightCocone
svn revert -R ../src/TightCocone
svn revert -R VolumeGridRover
svn revert -R ../src/VolumeGridRover
svn revert -R Filters
svn revert -R ../src/Filters
svn revert -R multi_sdf
svn revert -R ../src/multi_sdf
