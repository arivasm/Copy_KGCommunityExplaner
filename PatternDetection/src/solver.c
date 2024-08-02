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
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "dlist.h"
#include "graph_adj.h"
#include "memory.h"
#include "hash_iset.h"
#include "hash_map.h"
#include "util.h"
#include "semEP_Node.h"

/*********************************
 ** Constants
 *********************************/

#define NOCOLOR       -1
#define NONODE        -1
#define INFTY         INT_MAX
#define EPSILON       10e-7
#define NO_ETYPE      -1
#define HT_SIM_SZ     7
#define ONE           1.0001
#define NOSIM         -999.999999
#define COLOR_MARK    5000
#define ZERO          0

/************************************************************
 ** General macros
 ************************************************************/

#define CHECK_ELEM(e)						\
     do {							\
	  if ((e) == NOELEM)					\
	       fatal("Error in searching a valid element\n");	\
     } while(0)

/************************************************************
 ** Macros used for the binary heap implementation
 ************************************************************/

#define PARENT(i)     ((int)(((i) - 1) / 2))
#define LEFT(i)       (((i) * 2) + 1)
#define RIGHT(i)      (((i) * 2) + 2)

/*********************************
 ** Structures
 *********************************/

typedef struct saturation {
     int node;
     int n_adj_colors;
     struct hash_iset color_used;
} saturation_t;

typedef struct pqueue {
     int size;
     int *node_pos;
     saturation_t **heap;
} pqueue_t;

typedef struct info_partition {
     bool is_new_color;
     int color;
     double nc;
     double pairwise_sim;
     double cDensity;
     double sim;
} info_partition_t;

typedef struct sim_tuple {
     double total;
     double sim;
} sim_tuple_t;

/*************************************
 *************************************
 **
 ** Utilities 
 **
 ************************************
 ************************************/

static void init_clusters(clusters_t *c)
{
     c->partitions.alloc = 0;
     c->partitions.nr = 0.0;
     c->partitions.data = NULL;
}

/**
 *  Function to get the similirity between entities
 */
static inline double similarity(double **ematrix, unsigned int n,
				unsigned int pos1, unsigned int pos2)
{
     if ( (n <= pos1) || (n <= pos2) ) 
	  fatal("matrix similarity index out of bounds: pos1 = %u -- pos2 %u -- n %u", pos1, pos2, n);
     return ematrix[pos1][pos2];
}


static inline void init_info_partition(info_partition_t *ip)
{
     ip->is_new_color = false;
     ip->color = NOCOLOR;
     ip->nc = NOSIM;
     ip->pairwise_sim = NOSIM;
     ip->cDensity = NOSIM;
     ip->sim = NOSIM; 
}

static double get_density_average(clusters_t *c)
{
     unsigned i, n;
     double r;

     r = 0.0;
     n = c->partitions.nr;
     for (i = 0; i < n; i++) {
	  r += c->partitions.data[i]->cDensity;
     }
     return r/n;
}

#ifdef PRGDEBUG
static void print_coloring(clusters_t *clusters)
{
     struct color *c;
     unsigned int i, n;

     n = clusters->partitions.nr;
     printf("\nResults of the partitioning:\n");
     printf("Number of partitions: %u\n", n);
     printf("nc %.4f\n", clusters->nc);
     for (i = 0; i < n; i++) {
	  c = clusters->partitions.data[i];
	  assert((unsigned)c->id == i);
	  printf("id %d - cDensity %.4f - num. nodes %u\n",
		 c->id, c->cDensity, c->entities.nr);
     }
     printf("\n");
}

static inline bool eq_double(double x, double y)
{
     return (fabs(x - y) < 0.0001);
}
#endif

/*************************************
 *************************************
 **
 ** Graph to coloring
 **
 ************************************
 ************************************/

static void build_graph_from_nodes(const struct entity_array *ea,
				   struct graph_adj *gc)
{
     int cont;
     unsigned int j, k, i, n_nodes;
     bool find_similar_node;
     
     cont = 0;
     n_nodes = ea->nr;
     for (j = 0; j < n_nodes-1; j++) {
	  for (k = j+1; k < n_nodes; k++) {
	       if (MAX(ea->ematrix[j][k], ea->ematrix[k][j]) <= ea->threshold) {
		    find_similar_node = false;
		    for (i = 0; i < n_nodes; i++) {
			 if ((MAX(ea->ematrix[j][i], ea->ematrix[i][j]) > ea->threshold) &&
			     (MAX(ea->ematrix[k][i], ea->ematrix[i][k]) > ea->threshold)) {
			     find_similar_node = true;
			     break;
			 }
			 
		    }
		    if (!find_similar_node ) {
			 add_arc(gc, cont, j, k);
			 add_arc(gc, cont, k, j);
			 cont++;
		    }
	       }
	  }
     }
}

/*************************************
 *************************************
 **
 ** Checking and control
 **
 ************************************
 ************************************/

/**
 * Pairwise symilarity for elements of a same type
 */
static double pairwise_sim(const struct hash_iset *entities,
			   double **ematrix, unsigned int nr)
{
     double total_sim;
     unsigned int i, j, k, l, alloc, n;
     int a, b;
     int *data;
   
     data = entities->htable;
     n = entities->nr;
     alloc = entities->alloc;
     if (n == 0) {
	  total_sim = 0.0;
     } else if (n == 1) {
	  k = 0;
	  a = NOELEM;
	  for (k = 0; k < alloc; k++) {
	       if (data[k] != NOELEM) {
		    a = data[k];
		    break;
	       }
	  }
	  CHECK_ELEM(a);
	  total_sim = EPSILON;
     } else {
	  total_sim = 0.0;
	  k = 0;
	  i = 0;
	  while ((i < n-1) && (k < alloc)) {
	       if (data[k] != NOELEM) {
		    a = data[k];
		    j = i+1;
		    l = k+1; 
		    while ((j < n) && (l < alloc)) {
			 if (data[l] != NOELEM) {
			      b = data[l];
			      assert(a != b);
			      total_sim += similarity(ematrix, nr, a, b);
			      j++;
			 }
			 l++;
		    }
		    assert(j == n);
		    i++;
	       }
	       k++;
	  }
	  assert(i == (n-1)); 
     }
     return total_sim;
}

/**
 * Cluster density
 */
static double cDensity(struct color *c, struct entity_array *ea)
{
     double cdensity;
     unsigned int n;
     
     n = c->entities.nr;
     if (n == 0) {
	  error("Warning, processing a empty partition\n");
	  cdensity = 0.0;
     } else if (n == 1){
	  cdensity = EPSILON;
     } else {
	  cdensity = pairwise_sim(&c->entities, ea->ematrix, ea->nr)/((n*n-n)/2.0);
     }
     return cdensity;
}

#ifdef PRGDEBUG
static double color_density(const clusters_t *c, struct entity_array *ea)
{
     unsigned int i, n_colors;
     double nc;
     
     nc = 0.0;
     n_colors = c->partitions.nr;
     for (i = 0; i < n_colors; i++) {
	  nc += (i+1) - cDensity(c->partitions.data[i], ea);
     }
     return nc;
}
#endif

/*************************************
 *************************************
 **
 **  Priority Queue
 **
 ************************************
 ************************************/

static int compare_saturation(const struct graph_adj *g, const saturation_t *a,
				     const saturation_t *b)
{
     int r;

     r = 0;
     if (a->n_adj_colors > b->n_adj_colors) { 
	  r = 1;
     } else if (a->n_adj_colors < b->n_adj_colors) {
	  r = -1;
     } else {
	  if (g->degree[a->node].din > g->degree[b->node].din) {
	       r = 1;
	  } else if (g->degree[a->node].din < g->degree[b->node].din) { 
	       r = -1;
	  } else {
	       /* Numerical order */
	       if (a->node < b->node) {
		    r = 1;
	       } else if (a->node > b->node) {
		    r = -1;
	       } else {
		    fatal("Same pair the nodes in the comparison");
	       }
	  }
     }
     return r;
}

static void free_saturation_node(saturation_t *node)
{
     if (node) {
	  free_hash_iset(&node->color_used);
	  free(node);
     }
}

static void pq_insert(const struct graph_adj *g, pqueue_t *pq,
			     saturation_t *node)
{
     int i, p;
     saturation_t **tmp;

     tmp = xrealloc(pq->heap, (pq->size+1)*sizeof(saturation_t *));
     pq->heap = tmp;
     pq->heap[pq->size] = node;
     i = pq->size;
     p = PARENT(i);
     while((i > 0) &&  (compare_saturation(g, pq->heap[p], pq->heap[i]) < 0)){
	  SWAP(pq->heap[p], pq->heap[i]);
	  i = p;
	  p = PARENT(i);
     }
     pq->size++;
}

static int extract_max(const struct graph_adj *g, pqueue_t *pq, saturation_t **node)
{
     int i, j, l, r;
     saturation_t *aux;
     saturation_t **tmp;
          
     if(pq->size == 0)
	  return -1;

     *node = pq->heap[0];
     aux =  pq->heap[pq->size-1];
     SWAP(pq->node_pos[pq->heap[0]->node],
	  pq->node_pos[pq->heap[pq->size-1]->node]); /* SWAP the positions*/
     if((pq->size - 1) > 0){
	  tmp = (saturation_t **)xrealloc(pq->heap, (pq->size-1)*sizeof(saturation_t *));
	  pq->heap = tmp;
	  pq->size--;
     } else {
	  free(pq->heap);
	  pq->heap = NULL;
	  free(pq->node_pos);
	  pq->node_pos = NULL;
	  pq->size = 0;
	  return 0;
     }
     pq->heap[0] = aux;
     i = 0;
     while (true) {
	  l = LEFT(i);
	  r = RIGHT(i);
	  if((l < pq->size) && (compare_saturation(g, pq->heap[l], pq->heap[i]) > 0))
	       j = l;
	  else
	       j = i;

	  if((r < pq->size) && (compare_saturation(g, pq->heap[r], pq->heap[j]) > 0))
	       j = r;

	  if( j == i ) {
	       break;
	  } else {
	       SWAP(pq->node_pos[pq->heap[j]->node],
		    pq->node_pos[pq->heap[i]->node]); /* SWAP the positions*/
	       SWAP(pq->heap[j], pq->heap[i]);
	       i = j;
	  }
     }
     return 0;
}

static int increase_key(const struct graph_adj *g, pqueue_t *pq, int node, int color)
{
     int i, p, pos;
	  
     if (pq->size == 0)
	  return -1;

     pos = pq->node_pos[node];
     if (pos >= pq->size)
	  pos = -1;
   
     if (pos == -1)
	  return -2;

     if (insert_hash_iset(&(pq->heap[pos]->color_used), color))
	  pq->heap[pos]->n_adj_colors++;
     else
	  return 0;
     
     i = pos;
     p = PARENT(i);
     while((i > 0) && (compare_saturation(g, pq->heap[p], pq->heap[i]) < 0)){
	  SWAP(pq->node_pos[pq->heap[p]->node],
	       pq->node_pos[pq->heap[i]->node]); /* SWAP the positions*/
	  SWAP(pq->heap[p], pq->heap[i]);
	  i = p;
	  p = PARENT(i);
     }
     return 0;
}

static void pq_init(pqueue_t *pq)
{
     pq->size = 0;
     pq->heap = NULL;
     pq->node_pos = NULL;
}

static void init_saturation_pq(const struct graph_adj *g, pqueue_t *pq)
{
     int i, n;
     saturation_t *ns;
     
     n = g->n_nodes;
     for (i = 0; i < n; i++) {
	  ns = (saturation_t *)xmalloc(sizeof(saturation_t));
	  ns->node = i;
	  ns->n_adj_colors = 0;
	  init_hash_iset(&ns->color_used);
	  pq_insert(g, pq, ns);	  
     }
     assert(pq->size == n);
     pq->node_pos = (int *)xmalloc(n*sizeof(int));
     for (i = 0; i < n; i++) {
	  pq->node_pos[pq->heap[i]->node] = i;
     }
}

static inline void pq_delete(pqueue_t *pq)
{
     int i;
     
     for(i = 0; i < pq->size; i++)
	  free_saturation_node(pq->heap[i]);
     free(pq->node_pos);
     free(pq->heap);
}

/*************************************
 *************************************
 **
 **  Coloration Solver
 **
 ************************************
 ************************************/

static int get_color(const struct entity_array *node_color, int pos)
{
     int color;
   
     if (pos < 0)
	  fatal("Invalid position");
     if (node_color == NULL)
	  fatal("Invalid pointer to color");
     if((unsigned int)pos >= node_color->nr)
	  fatal("Error in color asignation"); 
     if (node_color->data[pos].cp == NULL)
	  color = NOCOLOR;
     else
	  color = node_color->data[pos].cp->id;

     return color;
}

static int greatest_saturation_node(const struct graph_adj *g, pqueue_t *pq, 
				    const struct entity_array *node_color)
{
     int r, color, node;
     saturation_t *ns;

     node = NONODE;
     ns = NULL;
     color = INFTY;
     r = extract_max(g, pq, &ns);

     if (r == -1)
	  fatal("No node without color");
     if (ns) {
	  color = get_color(node_color, ns->node);
	  node = ns->node;
     } else {
	  fatal("Error in get the greatest saturation node");
     }
     if (color != NOCOLOR)
	  fatal("Error in node to coloring");
#ifdef PRGDEBUG
     printf("Node %d; Num. of adjacent %d; Degree in %d\n", node, ns->n_adj_colors, g->degree[node].din);
#endif
     free_saturation_node(ns); 
     return node;
}

static inline struct color *new_color(int id, struct entity *e,
				      const struct entity_array *ea)   
{
     struct color *new;
	  
     new = xcalloc(1, sizeof(struct color));
     new->id = id;
     init_hash_iset(&new->entities);
     insert_hash_iset(&new->entities, e->id);
     new->sim_entities = pairwise_sim(&new->entities, ea->ematrix, ea->nr); 
     return new;
}

static void update_saturation_degree(const struct graph_adj *g,
				     pqueue_t *pq, int node,
				     struct entity_array *node_color)
{
     int r, color;
     struct arc_array adjs;
     unsigned int i, nr;
	  
     color = get_color(node_color, node);
     assert(color != NOCOLOR);
     adjs = get_adjacent_list(g, node);
     nr = adjs.nr;
     for (i = 0; i < nr; i++) {
	  r = increase_key(g, pq, adjs.data[i].to, color);
	  if (r == -1)
	       fatal("Error in update saturation degree\n");
     }
}

static bool *get_ady_used_color(const struct graph_adj *g,
				const struct entity_array *node_color, int node)
{
     struct arc_array adjs;
     bool *color_used;
     int color;
     size_t alloc;
     unsigned int i, nr;
     
     alloc = g->n_nodes*sizeof(bool);
     color_used = xmalloc(alloc);
     memset(color_used, false, alloc);
     adjs = get_adjacent_list(g, node);
     nr = adjs.nr;
     for (i = 0; i < nr; i++) {
	  color = get_color(node_color, adjs.data[i].to);
	  assert(color < g->n_nodes);
	  if (color != NOCOLOR) 
	       color_used[color] = true;
     }
     return color_used;
}

static void get_free_colors(const struct graph_adj *g,
			    const struct entity_array *solution,
			    int node, struct int_array *available_colors,
			    color_ptr_array_t *partitions)
{
     int cn;
     bool *color_used;
     struct color *ctmp;
     unsigned int i, n;
     
     n = partitions->nr;
     color_used = NULL;
     assert(g->n_nodes > node);
     assert(g->n_nodes >= (int)n+1); /* Number of colors not used is n+1 */
     color_used = get_ady_used_color(g, solution, node);
     cn = get_color(solution, node);
     if (cn != NOCOLOR) 
	  fatal("A adjacent node are using the same color");
     for (i = 0; i < n; i++) {
	  ctmp = partitions->data[i];
	  assert(ctmp->id == (int)i);
	  if (!color_used[i]) {
	       available_colors->data[available_colors->nr++] = i;
	  }
     }
     /* New color */
     if (available_colors->nr == 0)
	  available_colors->data[available_colors->nr++] = i;
     free(color_used);
}


static double get_sum_with_all_elements(const struct hash_iset *set,
					unsigned int e,
					double **ematrix, unsigned int nr)
{
     unsigned int i, n, k, alloc;
     int *data;
     double total;

     k = 0;
     i = 0;
     n = set->nr;
     data = set->htable;
     alloc = set->alloc;
     total = 0.0;
     while ((i < n) && (k < alloc)) {
	  if (data[k] != NOELEM) {
	       total += similarity(ematrix, nr, data[k], e);
	       i++;
	  }
	  k++;
     }
     assert(i == n);
     return total;
}

static sim_tuple_t agregate_similarity(double current_sim,
				       const struct hash_iset *set,
				       unsigned int e,
				       double **ematrix, unsigned int nr)
{
     unsigned int n;
     double sim, total;
     sim_tuple_t t;
   
     sim = 0.0;
     total = 0.0;
     n = set->nr;
     if (lookup_hash_iset(set, e)) {
	  fatal("Repeated node in the partition");
     } else {
	  total = get_sum_with_all_elements(set, e, ematrix, nr);
	  total += current_sim;
	  n++;
	  sim = total / ((n*n-n)/2.0); 
     }
     t.total = total;
     t.sim = sim;

     return t;
}

static info_partition_t density_with_new_node(const color_ptr_array_t *partitions,
					      const struct entity *new_node,
					      const struct entity_array *ea,
					      int color)
{
     struct color *cptr;
     info_partition_t ip;
     sim_tuple_t t1;
     double sim_type1;

     cptr = partitions->data[color];
     assert(cptr->id == color);
    
     sim_type1 = cptr->sim_entities;
     t1 = agregate_similarity(sim_type1, &cptr->entities,
			      new_node->id, ea->ematrix, ea->nr);
     ip.is_new_color = false;
     ip.color = color;
     ip.sim = t1.total;
     ip.cDensity = t1.sim;
     if (ip.cDensity >= ONE)
	  fatal("Error in the computation of the density of a partition %.3f", ip.cDensity);
     return ip;
}

static info_partition_t get_best_color(clusters_t *c, int new_node,
				       const struct int_array *free_colors,
				       struct entity_array *ea)
{
     int i, n, n_colors, curr_color;
     double nc_best, nc_new, nc_current;
     struct entity *nptr;
     info_partition_t ip_aux, ip_best;

     nc_new = 0;
     init_info_partition(&ip_best);
     n = free_colors->nr;
     nc_best = INFTY;
     n_colors = c->partitions.nr;
     nc_current = c->nc;
     nptr = &ea->data[new_node];
     for (i = 0; i < n; i++) {
	  curr_color = free_colors->data[i];
	  assert((curr_color >= 0) && (curr_color <= n_colors));
	  DEBUG("Color to evaluate %d ", curr_color);
	  if (n_colors == curr_color) {
	       /* We need to use a new color, then we have a partition with a element */
	       /* The density of a partition with a element is epsilon */
	       nc_new = nc_current + ((curr_color+1) - EPSILON);
	       ip_aux.is_new_color = true;
	       ip_aux.color = curr_color;
	       ip_aux.cDensity = EPSILON;
	       ip_aux.nc = nc_new; 
	  } else {
	       /* We coloring the new node with a used color */
	       ip_aux = density_with_new_node(&c->partitions, nptr, ea, curr_color);
	       nc_new = nc_current + c->partitions.data[curr_color]->cDensity - ip_aux.cDensity;
	       ip_aux.nc = nc_new;
	   }
	  if (nc_best > nc_new) {
	       nc_best = nc_new;
	       ip_best = ip_aux;
	  }
     }
     return ip_best;
}

static void set_colors(clusters_t *c, info_partition_t *ip, struct entity *nptr,
		       const struct entity_array *ea)
{
     struct color *cptr = NULL;
     
     if (ip->is_new_color) {
	  cptr = new_color(ip->color, nptr, ea);
	  cptr->cDensity = ip->cDensity;
     	  ARRAY_PUSH(c->partitions, cptr);
     } else {
	  cptr = c->partitions.data[ip->color];
	  cptr->cDensity = ip->cDensity;
	  cptr->sim_entities = ip->sim;
	  insert_hash_iset(&cptr->entities, nptr->id);
     }
     nptr->cp = cptr;
     c->nc = ip->nc;
}

/**
 * Clustering solver based on the coloring algorithm called DSATUR
 */ 
static void coloring(struct entity_array *ea, const struct graph_adj *g,
		     clusters_t *c)
{
     struct int_array free_colors;
     int colored_nodes, new_node, n;
     pqueue_t pq_saturation;
     struct entity *nptr;
     struct color *cptr;
     info_partition_t ip;
     
     init_struct_array(free_colors);
     colored_nodes = 0;
     ALLOC_GROW(free_colors.data, (unsigned int)g->n_nodes, free_colors.alloc);
     pq_init(&pq_saturation);
     init_saturation_pq(g, &pq_saturation);
     assert(c->partitions.nr == 0);
     /* We color a first node */
     new_node = greatest_saturation_node(g, &pq_saturation, ea);
     if (new_node == NONODE)
	  fatal("Error getting the greatest saturation node");
     nptr = &ea->data[new_node];
     assert(new_node == nptr->id);
     cptr = new_color(0, nptr, ea);
     cptr->cDensity = EPSILON;
     ARRAY_PUSH(c->partitions, cptr);
     c->nc = 1.0 - cDensity(cptr, ea); /* First color */
     assert(cDensity(cptr, ea) ==
	    cDensity(c->partitions.data[c->partitions.nr-1], ea));
     nptr->cp = cptr;
     colored_nodes++;
     if (pq_saturation.size != 0)
	  update_saturation_degree(g, &pq_saturation, new_node, ea);
     /* We will color all the nodes */
     n = g->n_nodes;
     while (colored_nodes < n) {
	  if ( (colored_nodes % COLOR_MARK) == 0)
	       printf("++++ Number of nodes colored so far: %d ++++\n", colored_nodes);
	  new_node = greatest_saturation_node(g, &pq_saturation, ea);
	  if (new_node == NONODE)
	       fatal("Error getting the greatest saturation node");
	  nptr = &ea->data[new_node];
	  assert(new_node == nptr->id);
	  free_colors.nr = 0;
	  get_free_colors(g, ea, new_node, &free_colors, &c->partitions);
#ifdef PRGDEBUG
	  printf("Free colors: ");
	  print_int_array(free_colors);
#endif
	  ip = get_best_color(c, new_node, &free_colors, ea);
	  if (ip.nc == NOSIM)
	       fatal("Best color is NULL");
	  set_colors(c, &ip, nptr, ea);
	  colored_nodes++;
	  if (pq_saturation.size != 0)
	       update_saturation_degree(g, &pq_saturation, new_node, ea);
#ifdef PRGDEBUG	  
	  if (!eq_double(color_density(c, entities), c->nc)) 
	       error("Different in the color density value %.3f %.3f\n",
		     color_density(c, ea), c->nc);
	  else
	       printf("Equal cDensity %.3f -- nc %.3f\n", color_density(c, ea), c->nc);
#endif
     }
     if (pq_saturation.size != 0)
	  fatal("Incomplete coloration\n");
     pq_delete(&pq_saturation);
     free(free_colors.data);
}

/*************************************
 *************************************
 **
 **  Get output files
 **
 ************************************
 ************************************/

static char *print_clustering(const struct color_ptr_array *partitions,
			      struct entity_array *ea,
			      const char *name)
{
     FILE *f;
     unsigned int i, j, n, nr, k, alloc;
     char *output1, *output2, *message;
     struct stat st;
     struct entity node1;
     struct color *cluster;
     int id_node1;
     int *data;
     time_t raw_time;
     struct tm * time_info;

     time(&raw_time);
     time_info = localtime(&raw_time);

    if (asprintf(&output1, "%s-%.4f-%dh-%dm-%ds-Clusters", name, ea->threshold, time_info->tm_hour, time_info->tm_min, time_info->tm_sec) == -1)
	  fatal("Error in output directory");

     if (stat(output1, &st) == -1)
	  mkdir(output1, 0700);
     else
	  fatal("The folder %s exists. Please, delete the folder and rerun SemEP-Node", output1);
     
     if (asprintf(&message, "Cluster directory: %s\n", output1) == -1)
	  fatal("Error in output message");

     printf("Number of partitions: %u\n", partitions->nr);
     n = partitions->nr;
     for (i = 0; i < n; i++) {
	  cluster = partitions->data[i];
	  if (asprintf(&output2, "%s/cluster-%u.txt", output1, i) == -1)
	       fatal("Error in cluster file");
          f = fopen(output2, "w");
	  if (!f)
	       fatal("No descriptor file specified, abort\n");
	  k = 0;
	  j = 0;
	  nr = cluster->entities.nr;
	  data = cluster->entities.htable;
	  alloc = cluster->entities.alloc;
	  while ((j < nr) && (k < alloc)) {
	       if (data[k] != NOELEM) {
		    id_node1 = data[k];
		    node1 = ea->data[id_node1];
		    fprintf(f ,"%s\n", node1.name);
		    j++;
	       }
	       k++;
	  }
	  assert(j == nr);
	  fclose(f);
	  free(output2);
     }
     free(output1);
     return message;
}

/*************************************
 *************************************
 **
 **  Freeing memory
 **
 ************************************
 ************************************/

static void free_entity_array(struct entity_array *e)
{
     unsigned int i, n;

     free_double_matrix(e->ematrix, 0, 0);
     n = e->nr;
     for (i = 0; i < n; i++)
	  if (e->data[i].name)
	       free(e->data[i].name);
     if (e->data)
	  free(e->data);
     e->threshold = 0.0;
     e->nr = 0;
     e->alloc = 0;
}

static void free_clusters(clusters_t *c)
{
     unsigned int i, n;
     struct color *ctmp;
     
     n = c->partitions.nr;
     for (i = 0; i < n; i++) {
	  ctmp = c->partitions.data[i];
	  free_hash_iset(&ctmp->entities);
	  free(ctmp);
     }
     free(c->partitions.data);
     c->partitions.nr = 0;
     c->partitions.alloc = 0;
}

/*************************************
 *************************************
 **
 **  semEP solver main function
 **
 ************************************
 ************************************/

void semEP_solver(struct entity_array *ea, const char *node_name)
{
     clusters_t *c;
     double density;
     struct graph_adj gc;
     clock_t ti, tf;
     char *message;
     unsigned int n_nodes;

     density = 0.0;
     c = (clusters_t *)xcalloc(1, sizeof(clusters_t));
     init_clusters(c);
     ti = clock();
     n_nodes = ea->nr;
     init_graph_adj(&gc, n_nodes);
     build_graph_from_nodes(ea, &gc);
     tf = clock();
     printf("Time to build the graph to coloring: %.4f secs\n",
	    (double)(tf-ti)/CLOCKS_PER_SEC);
     printf("Graph to coloring - Num. of Nodes: %d; Num. of Edges: %ld\n",
	    gc.n_nodes, gc.n_arcs/2);
#ifdef PRGDEBUG
     print_graph_adj(&gc);
#endif
     ti = clock();
     if (gc.n_nodes != 0) {
	  printf("Start coloring\n");
	  coloring(ea, &gc, c);
	  density = get_density_average(c);
	  printf("Average density of the partitions: %.4f \n", density);
#ifdef PRGDEBUG
	  density = color_density(c, ea);
	  if (!eq_double(density, c->nc)) {
	       print_coloring(c);
	       fatal("The density values do not match %.3f %.3f\n", density, c->nc);
	  }
#endif	  
     } else {
	  fatal("Graph to coloring has no nodes");
     }
     message = print_clustering(&c->partitions, ea, node_name);
     printf("%s", message);
     free(message);
     free_graph_adj(&gc);
     free_clusters(c);
     free(c);
     free_entity_array(ea);
     tf = clock();
     printf("Coloring solver time: %.4f secs\n", (double)(tf-ti)/CLOCKS_PER_SEC);
}
