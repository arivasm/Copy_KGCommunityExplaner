/**
 * Copyright (C) 2013-2017 Universidad Simón Bolívar
 * Copyright (C) 2017-2018 Research Center L3S
 *
 * Copying: GNU GENERAL PUBLIC LICENSE Version 2
 * @author Guillermo Palma <palma@l3s.de>
 */


#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "semEP_Node.h"
#include "util.h"
#include "hash_map.h"
#include "memory.h"

/*********************************
 ** Contants
 *********************************/

#define BUFSZ          256
#define MIN_HASH_SZ    389
#define ONE            1.0001
#define ZERO           0

/*********************************
 ** Structures
 *********************************/

struct char_array {
  unsigned nr;
  unsigned alloc;
  char *data;
};

struct concept {
     unsigned int pos;
     struct hash_entry entry;
};

/*************************************
 ** Utilities
 ************************************/

#ifdef PRGDEBUG
static void print_hash_term(struct hash_map *term_pos)
{
     struct concept *item;
     struct hash_entry *hentry;

     printf("\nMap term-position\n");
     printf("Number of terms: %u\n", term_pos->fill);
     hmap_for_each(hentry, term_pos) {
	  item = hash_entry(hentry, struct concept, entry);
	  printf("** %s %u\n", hentry->key, item->pos);
     }
}
#endif

static void free_entities_table(struct hash_map *term_pos)
{
     struct concept *item;
     struct hash_entry *hentry;
     struct hlist_node *n;

     hmap_for_each_safe(hentry, n, term_pos) {
	  item = hash_entry(hentry, struct concept, entry);
	  hmap_delete(term_pos, hentry);
	  free(item);
     }
     hmap_destroy(term_pos);
}

/*********************************
 ** String processing
 *********************************/

static inline void string_clean(struct char_array *buf)
{
     buf->nr = 0;
}

static inline void add_char(struct char_array *buf, char ch)
{
     unsigned alloc, nr;

     alloc = buf->alloc;
     nr = buf->nr;
     if (nr == alloc) {
	  alloc = BUFSZ + alloc;
	  buf->data = realloc(buf->data, alloc);
	  if (!buf->data) {
	       fprintf(stderr, "Out of memory, realloc failed\n");
	       exit(1);
	  }
	  buf->alloc = alloc;
     }
     buf->data[nr] = ch;
     buf->nr++;
}

static inline void init_char_array(struct char_array *buf)
{
     buf->data = calloc(BUFSZ, 1);
     if (!buf->data) {
	  fprintf(stderr, "Out of memory, calloc failed\n");
	  exit(1);
     }
     buf->alloc = BUFSZ;
     buf->nr = 0;
}

/*********************************
 ** Node processing
 *********************************/

static void nodes_load(const char *filename, struct hash_map *vertex_table,
		       struct entity_array *ea, double threshold)
{
     FILE *f;
     char buf[BUFSZ];
     size_t last, len;
     int i, n;
     int l;
     struct concept *item;
          
     ea->threshold = threshold; 
     printf("Proccessing the vertex file %s ...\n", filename);
     /* Processing of the file with the vertices */
     f = fopen(filename, "r");
     if (!f)
	  fatal("no terms file specified, abort\n");
     if (fgets(buf, sizeof(buf), f) == NULL)
	  fatal("error reading file");
     errno = 0;
     n = strtol(buf, NULL, 10);
     if (errno)
	  fatal("error in the conversion of string to integer\n");
     assert(ea->nr == 0);
     ALLOC_GROW(ea->data, (unsigned int)n, ea->alloc);
      for (i = 0; i < n; i++) {
	   /* Reading of the string */
	  if (fgets(buf, sizeof(buf), f) == NULL)
	       fatal("error reading file");

          /* Change of the last character */
	  len = strlen(buf);
	  last = len - 1;
	  if(buf[last] == '\n')
	       buf[last] = 0;

	  ea->data[ea->nr].id = i;
	  ea->data[ea->nr].cp = NULL;
	  l = asprintf(&ea->data[ea->nr].name,"%s", buf);
	  if (l == -1)
	       fatal("error in term copy");
	  ea->nr++;

          /* Copying the string to the hash table */
	  item = xmalloc(sizeof(struct concept));
	  item->pos = i;
	  if (hmap_add_if_not_member(vertex_table, &item->entry, buf, len) != NULL)
	       fatal("the term %s is repeated in the file %s\n", buf, filename);
     
     }
     fclose(f);
}

/*********************************
 ** Matrices processing
 *********************************/

static void similarity_matrix_load(const char *filename,
				   struct entity_array *ea,
				   unsigned int nodes)
{
     FILE *f;
     struct char_array buf;
     int n, i, j;
     int ch;
     double val;
     
     f = fopen(filename, "r");
     if (!f) {
	  fatal("No instance file specified, abort\n");
     }
     n = 0;
     init_char_array(&buf);
     ch = getc(f);
     errno = 0;
     /* read number of nodes and arcs */
     while((ch != '\n') && (ch != EOF)) {
	  add_char(&buf, ch);
	  ch = getc(f);
     }
     if (ch != EOF) {
	  add_char(&buf, '\0');
	  n = strtol(buf.data, NULL, 10);
	  if (errno)
	       fatal("error in the conversion of string to integer\n");

	  if ((unsigned)n != nodes ||  (n < 0))
	       fatal("Error, in the number of elements in the matrix");
	  
     } else {
	  fatal("error reading the matrix data file\n");
     }
     string_clean(&buf);
     ea->ematrix = double_matrix(0, n, 0, n);
     i = 0;
     j = 0;
     ch = getc(f);
     if (ch == EOF) {
	  fatal("error reading the matrix data file\n");
     }
     errno = 0;
     while (ch != EOF) {
	  if ((ch != ' ') && (ch != '\n')) {
	       add_char(&buf, ch);
	  } else {
	       add_char(&buf, '\0');
	       val = strtod(buf.data, NULL);
	       if (val >= ONE)
		    fatal("similarity value greater than one: %.5f", val);
	       else if (val < ZERO)
		    fatal("similarity value lower than zero: %.5f", val);
	       else
		    ea->ematrix[i][j] = val;
	       if (errno)
		    fatal("error in the conversion of string to double\n");
	       if (ch == ' ') {
		    j++;
	       } else if (ch == '\n') {
		    i++;
		    j = 0;
	       } else {
		    fatal("unknown character");
	       }
	       string_clean(&buf);
	  }
	  ch = getc(f);
     }
     fclose(f);
     free_array(buf);
}

/*********************************
** Getting the Problem Data
*********************************/

struct entity_array get_input_data(const char *nodes_filename,
				   const char *matrix_filename,
				   double threshold)
{
     struct hash_map vertex_table;
     struct entity_array ea;
            
     hmap_create(&vertex_table, MIN_HASH_SZ);
     init_entity_array(&ea);
     nodes_load(nodes_filename, &vertex_table, &ea, threshold);
     similarity_matrix_load(matrix_filename, &ea, ea.nr);
#ifdef PRGDEBUG
     print_hash_term(&vertex_table);
     print_matrix(ea.ematrix, ea.nr);
#endif
      free_entities_table(&vertex_table);
     return ea;
}
