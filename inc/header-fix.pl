#!/usr/bin/perl

# Joe R. :: transfix@ices.utexas.edu :: 01/29/2010
# The purpose of this script is to massage our old project build methodologies towards
# sanity by forcing include statements to have their module names prefixed in order to avoid
# conflicts with libraries sharing the same header filenames.  It is also useful to explicitly
# show the module dependencies of each cpp file including headers of CVC libraries.
#
# For example:
#   #include "geoframe.h"
#   becomes
#   #include <LBIE_lib/geoframe.h>
#
# This program should be run from either the inc directory of the CVC standard
# project layout. See the following:
# http://cvcweb.ices.utexas.edu/mediawiki/index.php/Standard_CVC_Project_Template
#
# Example run of header-fix.pl printing out results of the search for every file.
# Assumes $do_modify has been set to 0
# $ cd $PROJECT_ROOT
# $ cd inc
# $ find . -name "*.h" -exec perl $SCRIPT_LOCATION/header-fix.pl \{\} \; -print
# $ find ../src -name "*.cpp" -exec perl $SCRIPT_LOCATION/header-fix.pl \{\} \; -print
#
# Example run of header-fix.pl using find to iterate across all 
# headers and sources in a project, replacing header declarations as needed.
# Assumes $do_modify has been set to 1.
# Note the -i option for perl, that allows edits of files in place:
# $ cd $PROJECT_ROOT
# $ cd inc
# $ find . -name "*.h" -exec perl -i $SCRIPT_LOCATION/header-fix.pl \{\} \; -print
# $ find ../src -name "*.cpp" -exec perl -i $SCRIPT_LOCATION/header-fix.pl \{\} \; -print

# ---- Change History ----
# 02/19/2010 -- Joe R. -- Added more commentary.

# If do_modify is non-zero, the program will print out a modified version of the
# input file.  Else it will simply print out 1 line for each include statement in
# the input file listing the input file itself, the bare include filename, a list
# of candidate include files that share the same filename, and the chosen header
# in the list (with a preference for headers that share the same module as the
# input file)
$do_modify = 1;

# only 1 input file for each run of header-fix.pl at a time!
$inputfile = $ARGV[0];
#removing leading ./
$inputfile =~ s/^\.\///g;

while(<>)
{
	#extract included header file string
	if(m/^\s*#include\s*("|<)(\S+)("|>).*$/)
	{	
		#extract just the included header filename then
		#search for all files with that name
		@path_elms = split(/\//,$2);
		$filename = $path_elms[$#path_elms];
		$locs = `find . -name "$filename"`;

		#get each individual file as array elements
		@locs = split(/\n/,$locs);

		#remove the leading in each found file ./ if there is any
		foreach $i (0 .. $#locs)
		{
			$locs[$i] =~ s/^\.\///g;
		}

		#rejoin them back together for printing
		$locs = join(" : ",@locs);

		#if nothing found, leave the line as it is
		if(length($locs) == 0)
		{
			if($do_modify)
			{
				print;
			}
			next;
		}
		
		#get the module that the input file is part of
		@path_elms = split(/\//,$inputfile);
		@mod_elms = @path_elms;
		pop(@mod_elms);
		$mod_elms = join("/",@mod_elms);

		#Now search each candidate filename for a matching module.
		#We preferr to use the header from the same module as the
		#current input file.  If none is found, use the first in the list.
		$chosen_header = $locs[0];
		foreach $i (0 .. $#locs)
		{
			@path_elms = split(/\//,$locs[$i]);
			@cur_mod_elms = @path_elms;
			pop(@cur_mod_elms);
			$cur_mod_elms = join("/",@cur_mod_elms);

			#if($mod_elms eq $cur_mod_elms)
			if($mod_elms =~ m/$cur_mod_elms/)
			{
				$chosen_header = $locs[$i];
			}
		}	

		if($do_modify)
		{	
			print "#include <$chosen_header>\n";
		}
		else
		{
			print "$inputfile : $filename :: $locs :::: $chosen_header\n";
		}
	}
	elsif($do_modify)
	{
		print;
	}
}

