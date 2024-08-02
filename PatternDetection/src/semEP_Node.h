/**
 * Copyright (C) 2012-2017 Universidad Simón Bolívar
 * Copyright (C) 2018 Research Center L3S
 *
 * Copying: GNU GENERAL PUBLIC LICENSE Version 2
 * @author Guillermo Palma <palma@l3s.de>
 */

#ifndef ___SEMEP_NODE_H
#define ___SEMEP_NODE_H

#include "hash_iset.h"
#include "hash_map.h"

#define BUFSZ     256
#define NO_ETYPE  -1

struct int_array {
     unsigned int nr;
     unsigned int alloc;
     int *data;
};

struct entity {
     int id;
     char *name;
     struct color *cp;
};

struct entity_array {
     unsigned int nr;
     unsigned int alloc;
     double threshold;
     double **ematrix;
     struct entity *data;
};

struct color {
     int id;
     double cDensity;
     double sim_entities;
     struct hash_iset entities;
};

typedef struct color_ptr_array {
     unsigned nr;
     unsigned alloc;
     struct color **data;
} color_ptr_array_t;

typedef struct clusters {
     double nc;                    /* Value to minimizing */
     color_ptr_array_t partitions; /* Array with the clustes */
} clusters_t;

static inline void init_entity_array(struct entity_array *e)
{
     e->nr = 0;
     e->alloc = 0;
     e->threshold = 0.0;
     e->ematrix = NULL;
     e->data = NULL;
}

struct entity_array get_input_data(const char *nodes_filename,
				   const char *matrix_filename,
				   double threshold);

void semEP_solver(struct entity_array *entities, const char *node_name);

#endif /* ___SEMEP_NODE_H */
