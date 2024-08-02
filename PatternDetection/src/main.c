/**
 * Copyright (C) 2012-2017 Universidad Simón Bolívar
 * Copyright (C) 2018 Research Center L3S
 * 
 * Copying: GNU GENERAL PUBLIC LICENSE Version 2
 * @author Guillermo Palma <palma@l3s.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <assert.h>

#include "util.h"
#include "memory.h"
#include "semEP_Node.h"

#define N_ARGS   3

static char *nodes_filename;
static char *matrix_filename;
static double threshold;

/*********************************
 **  Parse Arguments
 *********************************/

static void display_usage(void)
{
  fatal("Incorrect arguments:\n\tsemEP-node <nodes> <similarity matrix> <threshold>\n");
}

static void get_name(const char *filename, int n, char *instance)
{
  char buf[n];
  char *aux = NULL;
  
  strcpy(buf,filename);
  aux = strtok(buf,"/");
  do {
    strcpy(instance, aux);
  } while((aux = strtok(NULL,"/")) != NULL);
  
  strcpy(buf,instance);
  aux = strtok(buf,".");
  strcpy(instance, aux);
}

static void print_args(void)
{
  printf("\n**********************************************\n");
  printf("Parameters:\n");
  printf("Nodes file name: %s\n", nodes_filename);
  printf("Matrix file name: %s\n", matrix_filename);
  printf("Threshold: %.3f\n", threshold);
  printf("************************************************\n");
}

static void parse_args(int argc, char **argv)
{
  int i;
  
  if (argc != 4) 
    display_usage();
  i = 1;
  nodes_filename = argv[i++];
  matrix_filename = argv[i++];
  threshold = strtod(argv[i++], (char **)NULL);
}

/*********************************
 *********************************
 **
 **       Main section
 **
 *********************************
 **********************************/

int main(int argc, char **argv)
{
  int len;
  clock_t ti, tf;
  static char *name;
  struct entity_array ea;
  
  ti = clock();
  parse_args(argc, argv);
  print_args();
  len = strlen(nodes_filename) + 1;
  name = xcalloc(len, 1);
  get_name(nodes_filename, len, name);
  printf("\n**** GO semEP-Node! **** \n");
  ea = get_input_data(nodes_filename, matrix_filename, threshold);
  printf("Number of nodes of the input file: %u\n", ea.nr);
  semEP_solver(&ea, name);
  free(name);
  tf = clock();
  printf("Total time %.3f secs\n", (double)(tf-ti)/CLOCKS_PER_SEC);
  return 0;
}
