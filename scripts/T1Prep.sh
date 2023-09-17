#! /bin/sh
#
# PURPOSE:
#
# USAGE:
#
# INPUT:
#
# OUTPUT:
#
# FUNCTIONS:
#

########################################################
# global parameters
########################################################
version='T1Prep.sh, v 1.00 2003/04/24 Christian Gaser'

main ()
{
  parse_args ${1+"$@"}
  process

  exit 0
}


parse_args ()
{
  local optname optarg

  while [ $# -gt 0 ]
  do
    optname="`echo $1 | sed 's,=.*,,'`"
    optarg="`echo $1 | sed 's,^[^=]*=,,'`"
    case "$1" in
        --columns=* | --cols=*)
            optarg=`echo $optarg | sed 's,[^0-9],,g'`
            exit_if_empty "$optname" "$optarg"
            GLOBAL_columns=$optarg
            ;;
        --one-index-page* | --one-index*)
            GLOBAL_single_index_page=1
            ;;
        --quiet | -q)
            GLOBAL_show_progress=0
            ;;
        -h | --help | -v | --version | -V)
            help
            exit 1
            ;;
        -*)
            echo "`basename $0`: ERROR: Unrecognized option \"$1\"" >&2
            ;;
        *)
            ARGV_list="$ARGV_list $1"
            ;;
    esac
    shift
  done

}

exit_if_empty ()
{
  local desc val

  desc="$1"
  shift
  val="$*"

  if [ -z "$val" ]
  then
    echo ERROR: No argument given with \"$desc\" command line argument! >&2
    exit 1
  fi
}

process ()
{
  if [ -n "$ARGV_list" ]
  then
    for i in $ARGV_list
    do
        echo $i
    done
  else
    echo "No arguments given."
    help
    exit 1
  fi

}

########################################################
# help
########################################################

help ()
{
cat <<__EOM__

USAGE:

PURPOSE:

INPUT:

OUTPUT:

USED FUNCTIONS:

This script was written by Christian Gaser (christian.gaser@uni-jena.de).
This is version ${version}.

__EOM__
}

########################################################
# call main program
########################################################

main ${1+"$@"}
