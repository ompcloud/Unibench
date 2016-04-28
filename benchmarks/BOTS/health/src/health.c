/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/**********************************************************************************************/

/* OLDEN parallel C for dynamic structures: compiler, runtime system
 * and benchmarks
 *
 * Copyright (C) 1994-1996 by Anne Rogers (amr@cs.princeton.edu) and
 * Martin Carlisle (mcc@cs.princeton.edu)
 * ALL RIGHTS RESERVED.
 *
 * OLDEN is distributed under the following conditions:
 *
 * You may make copies of OLDEN for your own use and modify those copies.
 *
 * All copies of OLDEN must retain our names and copyright notice.
 *
 * You may not sell OLDEN or distribute OLDEN in conjunction with a
 * commercial product or service without the expressed written consent of
 * Anne Rogers and Martin Carlisle.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE.
 *
 */


/*******************************************************************
 *  Health.c : Model of the Colombian Health Care System           *
 *******************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include "health.h"
#include "../../common/BOTSCommonUtils.h"
/* global variables */
int sim_level;
int sim_cities;
int aba;
int sim_population_ratio;
int sim_time;
int sim_assess_time;
int sim_convalescence_time;
int32_t sim_seed;
float sim_get_sick_p;
float sim_convalescence_p;
float sim_realloc_p;
int sim_pid = 0;

int res_population;
int res_hospitals;
int res_personnel;
int res_checkin;
int res_village;
int res_waiting;
int res_assess;
int res_inside;
float res_avg_stay;

int res_population_seq;
int res_hospitals_seq;
int res_personnel_seq;
int res_checkin_seq;
int res_village_seq;
int res_waiting_seq;
int res_assess_seq;
int res_inside_seq;
float res_avg_stay_seq;

int cutoff_value = 2;
int manual_cutoff, if_cutoff;

/**********************************************************
 * Handles math routines for health.c                     *
 **********************************************************/
float my_rand(int32_t *seed)
{
   int32_t k;
   int32_t idum = *seed;

   idum ^= MASK;
   k = idum / IQ;
   idum = IA * (idum - k * IQ) - IR * k;
   idum ^= MASK;
   if (idum < 0) idum  += IM;
   *seed = idum * IM;
   return (float) AM * idum;
}
/********************************************************************
 * Handles lists.                                                   *
 ********************************************************************/
void addList(struct Patient **list, struct Patient *patient)
{
   if (*list == NULL)
   {
      *list = patient;
      patient->back = NULL;
      patient->forward = NULL;
   }
   else
   {
      struct Patient *aux = *list;
      while (aux->forward != NULL) aux = aux->forward;
      aux->forward = patient;
      patient->back = aux;
      patient->forward = NULL;
   }
}
void removeList(struct Patient **list, struct Patient *patient)
{
#if 0
   struct Patient *aux = *list;

   if (patient == NULL) return;
   while((aux != NULL) && (aux != patient)) aux = aux->forward;

   // Patient not found
   if (aux == NULL) return;

   // Removing patient
   if (aux->back != NULL) aux->back->forward = aux->forward;
   else *list = aux->forward;
   if (aux->forward != NULL) aux->forward->back = aux->back;
#else
   if (patient->back != NULL) patient->back->forward = patient->forward;
   else *list = patient->forward;
   if (patient->forward != NULL) patient->forward->back = patient->back;
#endif
}
/**********************************************************************/
void allocate_village( struct Village **capital, struct Village *back,
   struct Village *next, int level, int32_t vid)
{
   int i, population, personnel;
   struct Village *current, *inext;
   struct Patient *patient;

   if (level == 0) *capital = NULL;
   else
   {
      personnel = (int) pow(2, level);
      population = personnel * sim_population_ratio;
      /* Allocate Village */
      *capital = (struct Village *) malloc(sizeof(struct Village));
      /* Initialize Village */
      (*capital)->back  = back;
      (*capital)->next  = next;
      (*capital)->level = level;
      (*capital)->id    = vid;
      (*capital)->seed  = vid * (IQ + sim_seed);
      (*capital)->population = NULL;
      for(i=0;i<population;i++)
      {
         patient = (struct Patient *)malloc(sizeof(struct Patient));
         patient->id = sim_pid++;
         patient->seed = (*capital)->seed;
         // changes seed for capital:
         my_rand(&((*capital)->seed));
         patient->hosps_visited = 0;
         patient->time          = 0;
         patient->time_left     = 0;
         patient->home_village = *capital;
         addList(&((*capital)->population), patient);
      }
      /* Initialize Hospital */
      (*capital)->hosp.personnel = personnel;
      (*capital)->hosp.free_personnel = personnel;
      (*capital)->hosp.assess = NULL;
      (*capital)->hosp.waiting = NULL;
      (*capital)->hosp.inside = NULL;
      (*capital)->hosp.realloc = NULL;
      omp_init_lock(&(*capital)->hosp.realloc_lock);
      // Create Cities (lower level)
      inext = NULL;
      for (i = sim_cities; i>0; i--)
      {
         allocate_village(&current, *capital, inext, level-1, (vid * (int32_t) sim_cities)+ (int32_t) i);
         inext = current;
      }
      (*capital)->forward = current;
   }
}
/**********************************************************************/
struct Results get_results(struct Village *village)
{
   struct Village *vlist;
   struct Patient *p;
   struct Results t_res, p_res;

   t_res.hosps_number     = 0.0;
   t_res.hosps_personnel  = 0.0;
   t_res.total_patients   = 0.0;
   t_res.total_in_village = 0.0;
   t_res.total_waiting    = 0.0;
   t_res.total_assess     = 0.0;
   t_res.total_inside     = 0.0;
   t_res.total_hosps_v    = 0.0;
   t_res.total_time       = 0.0;

   if (village == NULL) return t_res;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
      p_res = get_results(vlist);
      t_res.hosps_number     += p_res.hosps_number;
      t_res.hosps_personnel  += p_res.hosps_personnel;
      t_res.total_patients   += p_res.total_patients;
      t_res.total_in_village += p_res.total_in_village;
      t_res.total_waiting    += p_res.total_waiting;
      t_res.total_assess     += p_res.total_assess;
      t_res.total_inside     += p_res.total_inside;
      t_res.total_hosps_v    += p_res.total_hosps_v;
      t_res.total_time       += p_res.total_time;
      vlist = vlist->next;
   }
   t_res.hosps_number     += 1.0;
   t_res.hosps_personnel  += village->hosp.personnel;

   // Patients in the village
   p = village->population;
   while (p != NULL)
   {
      t_res.total_patients   += 1.0;
      t_res.total_in_village += 1.0;
      t_res.total_hosps_v    += (float)(p->hosps_visited);
      t_res.total_time       += (float)(p->time);
      p = p->forward;
   }
   // Patients in hospital: waiting
   p = village->hosp.waiting;
   while (p != NULL)
   {
      t_res.total_patients += 1.0;
      t_res.total_waiting  += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time);
      p = p->forward;
   }
   // Patients in hospital: assess
   p = village->hosp.assess;
   while (p != NULL)
   {
      t_res.total_patients += 1.0;
      t_res.total_assess   += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time);
      p = p->forward;
   }
   // Patients in hospital: inside
   p = village->hosp.inside;
   while (p != NULL)
   {
      t_res.total_patients += 1.0;
      t_res.total_inside   += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time);
      p = p->forward;
   }

   return t_res;
}
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
void check_patients_inside(struct Village *village)
{
   struct Patient *list = village->hosp.inside;
   struct Patient *p;

   while (list != NULL)
   {
      p = list;
      list = list->forward;
      p->time_left--;
      if (p->time_left == 0)
      {
         village->hosp.free_personnel++;
         removeList(&(village->hosp.inside), p);
         addList(&(village->population), p);
      }
   }
}
/**********************************************************************/
void check_patients_assess_par(struct Village *village)
{
   struct Patient *list = village->hosp.assess;
   float rand;
   struct Patient *p;

   while (list != NULL)
   {
      p = list;
      list = list->forward;
      p->time_left--;

      if (p->time_left == 0)
      {
         rand = my_rand(&(p->seed));
         /* sim_covalescense_p % */
         if (rand < sim_convalescence_p)
         {
            rand = my_rand(&(p->seed));
            /* !sim_realloc_p % or root hospital */
            if (rand > sim_realloc_p || village->level == sim_level)
            {
               removeList(&(village->hosp.assess), p);
               addList(&(village->hosp.inside), p);
               p->time_left = sim_convalescence_time;
               p->time += p->time_left;
            }
            else /* move to upper level hospital !!! */
            {
               village->hosp.free_personnel++;
               removeList(&(village->hosp.assess), p);
               omp_set_lock(&(village->hosp.realloc_lock));
               addList(&(village->back->hosp.realloc), p);
               omp_unset_lock(&(village->hosp.realloc_lock));
            }
         }
         else /* move to village */
         {
            village->hosp.free_personnel++;
            removeList(&(village->hosp.assess), p);
            addList(&(village->population), p);
         }
      }
   }
}
/**********************************************************************/
void check_patients_waiting(struct Village *village)
{
   struct Patient *list = village->hosp.waiting;
   struct Patient *p;

   while (list != NULL)
   {
      p = list;
      list = list->forward;
      if (village->hosp.free_personnel > 0)
      {
         village->hosp.free_personnel--;
         p->time_left = sim_assess_time;
         p->time += p->time_left;
         removeList(&(village->hosp.waiting), p);
         addList(&(village->hosp.assess), p);
      }
      else
      {
         p->time++;
      }
   }
}
/**********************************************************************/
void check_patients_realloc(struct Village *village)
{
   struct Patient *p, *s;

   while (village->hosp.realloc != NULL)
   {
      p = s = village->hosp.realloc;
      while (p != NULL)
      {
         if (p->id < s->id) s = p;
         p = p->forward;
      }
      removeList(&(village->hosp.realloc), s);
      put_in_hosp(&(village->hosp), s);
   }
}
/**********************************************************************/
void check_patients_population(struct Village *village)
{
   struct Patient *list = village->population;
   struct Patient *p;
   float rand;

   while (list != NULL)
   {
      p = list;
      list = list->forward;
      /* randomize in patient */
      rand = my_rand(&(p->seed));
      if (rand < sim_get_sick_p)
      {
         removeList(&(village->population), p);
         put_in_hosp(&(village->hosp), p);
      }
   }

}
/**********************************************************************/
void put_in_hosp(struct Hosp *hosp, struct Patient *patient)
{
   (patient->hosps_visited)++;

   if (hosp->free_personnel > 0)
   {
      hosp->free_personnel--;
      addList(&(hosp->assess), patient);
      patient->time_left = sim_assess_time;
      patient->time += patient->time_left;
   }
   else
   {
      addList(&(hosp->waiting), patient);
   }
}
/**********************************************************************/
void sim_village_par_if(struct Village *village)
{
   struct Village *vlist;

   // lowest level returns nothing
   // only for sim_village first call with village = NULL
   // recursive call cannot occurs
   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
#pragma omp task untied if((sim_level - village->level) < cutoff_value)
      sim_village_par_if(vlist);
      vlist = vlist->next;
   }

   /* Uses lists v->hosp->inside, and v->return */
   check_patients_inside(village);

   /* Uses lists v->hosp->assess, v->hosp->inside, v->population and (v->back->hosp->realloc) !!! */
   check_patients_assess_par(village);

   /* Uses lists v->hosp->waiting, and v->hosp->assess */
   check_patients_waiting(village);

#pragma omp taskwait

   /* Uses lists v->hosp->realloc, v->hosp->asses and v->hosp->waiting */
   check_patients_realloc(village);

   /* Uses list v->population, v->hosp->asses and v->h->waiting */
   check_patients_population(village);
}

void sim_village_par_manual(struct Village *village)
{
   struct Village *vlist;

   // lowest level returns nothing
   // only for sim_village first call with village = NULL
   // recursive call cannot occurs
   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   if ((sim_level-village->level) < cutoff_value)
   {
      while(vlist)
      {
#pragma omp task untied
         sim_village_par_manual(vlist);
         vlist = vlist->next;
      }
   }
   else
   {
      while(vlist)
      {
         sim_village_par_manual(vlist);
         vlist = vlist->next;
      }
   }

   /* Uses lists v->hosp->inside, and v->return */
   check_patients_inside(village);

   /* Uses lists v->hosp->assess, v->hosp->inside, v->population and (v->back->hosp->realloc) !!! */
   check_patients_assess_par(village);

   /* Uses lists v->hosp->waiting, and v->hosp->assess */
   check_patients_waiting(village);

   if ((sim_level-village->level) < cutoff_value)
   {
#pragma omp taskwait
   }

   /* Uses lists v->hosp->realloc, v->hosp->asses and v->hosp->waiting */
   check_patients_realloc(village);

   /* Uses list v->population, v->hosp->asses and v->h->waiting */
   check_patients_population(village);
}

void sim_village_par(struct Village *village)
{
   struct Village *vlist;

   // lowest level returns nothing
   // only for sim_village first call with village = NULL
   // recursive call cannot occurs
   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
#pragma omp task untied
      sim_village_par(vlist);
      vlist = vlist->next;
   }

   /* Uses lists v->hosp->inside, and v->return */
   check_patients_inside(village);

   /* Uses lists v->hosp->assess, v->hosp->inside, v->population and (v->back->hosp->realloc) !!! */
   check_patients_assess_par(village);

   /* Uses lists v->hosp->waiting, and v->hosp->assess */
   check_patients_waiting(village);

#pragma omp taskwait

   /* Uses lists v->hosp->realloc, v->hosp->asses and v->hosp->waiting */
   check_patients_realloc(village);

   /* Uses list v->population, v->hosp->asses and v->h->waiting */
   check_patients_population(village);
}


void sim_village_seq(struct Village *village)
{
   struct Village *vlist;

   // lowest level returns nothing
   // only for sim_village first call with village = NULL
   // recursive call cannot occurs
   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
      sim_village_par(vlist);
      vlist = vlist->next;
   }

   /* Uses lists v->hosp->inside, and v->return */
   check_patients_inside(village);

   /* Uses lists v->hosp->assess, v->hosp->inside, v->population and (v->back->hosp->realloc) !!! */
   check_patients_assess_par(village);

   /* Uses lists v->hosp->waiting, and v->hosp->assess */
   check_patients_waiting(village);

   /* Uses lists v->hosp->realloc, v->hosp->asses and v->hosp->waiting */
   check_patients_realloc(village);

   /* Uses list v->population, v->hosp->asses and v->h->waiting */
   check_patients_population(village);
}

/**********************************************************************/
void my_print(struct Village *village)
{
   struct Village *vlist;
   struct Patient *plist;

   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist) {
      my_print(vlist);
      vlist = vlist->next;
   }

   plist = village->population;

   while (plist != NULL) {
      //uncomment to debug properly
      //fprintf(stdout,"[pid:%d]",plist->id);
      plist = plist->forward;
   }
   //fprintf(stdout,"[vid:%d]\n",village->id);

}
/**********************************************************************/

void read_input_data_seq(char *filename)
{
   FILE *fin;
   int res;

   if ((fin = fopen(filename, "r")) == NULL) {
      fprintf(stdout,"Could not open sequence file (%s)\n", filename);
      exit (-1);
   }
   res = fscanf(fin,"%d %d %d %d %d %d %ld %f %f %f %d %d %d %d %d %d %d %d %f",
             &sim_level,
             &sim_cities,
             &sim_population_ratio,
             &aba,
             &sim_assess_time,
             &sim_convalescence_time,
             &sim_seed,
             &sim_get_sick_p,
             &sim_convalescence_p,
             &sim_realloc_p,
             &res_population_seq,
             &res_hospitals_seq,
             &res_personnel_seq,
             &res_checkin_seq,
             &res_village_seq,
             &res_waiting_seq,
             &res_assess_seq,
             &res_inside_seq,
             &res_avg_stay_seq
   );
   if ( res == EOF ) {
      fprintf(stdout,"Bogus input file (%s)\n", filename);
      exit(-1);
   }
   fclose(fin);

      // Printing input data
   fprintf(stdout,"\n");
   fprintf(stdout,"Number of levels    = %d\n", (int) sim_level);
   fprintf(stdout,"Cities per level    = %d\n", (int) sim_cities);
   fprintf(stdout,"Population ratio    = %d\n", (int) sim_population_ratio);
   fprintf(stdout,"Simulation time     = %d\n", (int) aba);
   fprintf(stdout,"Assess time         = %d\n", (int) sim_assess_time);
   fprintf(stdout,"Convalescence time  = %d\n", (int) sim_convalescence_time);
   fprintf(stdout,"Initial seed        = %d\n", (int) sim_seed);
   fprintf(stdout,"Get sick prob.      = %f\n", (float) sim_get_sick_p);
   fprintf(stdout,"Convalescence prob. = %f\n", (float) sim_convalescence_p);
   fprintf(stdout,"Realloc prob.       = %f\n", (float) sim_realloc_p);
}

void read_input_data_par(char *filename)
{
   FILE *fin;
   int res;

   if ((fin = fopen(filename, "r")) == NULL) {
      fprintf(stdout,"Could not open sequence file (%s)\n", filename);
      exit (-1);
   }
   res = fscanf(fin,"%d %d %d %d %d %d %ld %f %f %f %d %d %d %d %d %d %d %d %f",
             &sim_level,
             &sim_cities,
             &sim_population_ratio,
             &aba,
             &sim_assess_time,
             &sim_convalescence_time,
             &sim_seed,
             &sim_get_sick_p,
             &sim_convalescence_p,
             &sim_realloc_p,
             &res_population,
             &res_hospitals,
             &res_personnel,
             &res_checkin,
             &res_village,
             &res_waiting,
             &res_assess,
             &res_inside,
             &res_avg_stay
   );
   if ( res == EOF ) {
      fprintf(stdout,"Bogus input file (%s)\n", filename);
      exit(-1);
   }
   fclose(fin);

      // Printing input data
   fprintf(stdout,"\n");
   fprintf(stdout,"Number of levels    = %d\n", (int) sim_level);
   fprintf(stdout,"Cities per level    = %d\n", (int) sim_cities);
   fprintf(stdout,"Population ratio    = %d\n", (int) sim_population_ratio);
   fprintf(stdout,"Simulation time     = %d\n", (int) aba);
   fprintf(stdout,"Assess time         = %d\n", (int) sim_assess_time);
   fprintf(stdout,"Convalescence time  = %d\n", (int) sim_convalescence_time);
   fprintf(stdout,"Initial seed        = %d\n", (int) sim_seed);
   fprintf(stdout,"Get sick prob.      = %f\n", (float) sim_get_sick_p);
   fprintf(stdout,"Convalescence prob. = %f\n", (float) sim_convalescence_p);
   fprintf(stdout,"Realloc prob.       = %f\n", (float) sim_realloc_p);
}
int check_village_par(struct Village *top)
{
   struct Results result = get_results(top);
   int answer = 1;

   if (res_population != result.total_patients) answer = 0;
   if (res_hospitals != result.hosps_number) answer = 0;
   if (res_personnel != result.hosps_personnel) answer = 0;
   if (res_checkin != result.total_hosps_v) answer = 0;
   if (res_village != result.total_in_village) answer = 0;
   if (res_waiting != result.total_waiting) answer = 0;
   if (res_assess != result.total_assess) answer = 0;
   if (res_inside != result.total_inside) answer = 0;

   fprintf(stdout,"\n");
   fprintf(stdout,"Sim. Variables      = expect / result\n");
   fprintf(stdout,"Total population    = %6d / %6d people\n", (int)   res_population, (int) result.total_patients);
   fprintf(stdout,"Hospitals           = %6d / %6d people\n", (int)   res_hospitals, (int) result.hosps_number);
   fprintf(stdout,"Personnel           = %6d / %6d people\n", (int)   res_personnel, (int) result.hosps_personnel);
   fprintf(stdout,"Check-in's          = %6d / %6d people\n", (int)   res_checkin, (int) result.total_hosps_v);
   fprintf(stdout,"In Villages         = %6d / %6d people\n", (int)   res_village, (int) result.total_in_village);
   fprintf(stdout,"In Waiting List     = %6d / %6d people\n", (int)   res_waiting, (int) result.total_waiting);
   fprintf(stdout,"In Assess           = %6d / %6d people\n", (int)   res_assess, (int) result.total_assess);
   fprintf(stdout,"Inside Hospital     = %6d / %6d people\n", (int)   res_inside, (int) result.total_inside);
   fprintf(stdout,"Average Stay        = %6f / %6f u/time\n", (float) res_avg_stay,(float) result.total_time/result.total_patients);

   my_print(top);

   return answer;
}

int check_village_seq(struct Village *top)
{
   struct Results result = get_results(top);
   int answer = 1;

   if (res_population_seq != result.total_patients) answer = 0;
   if (res_hospitals_seq != result.hosps_number) answer = 0;
   if (res_personnel_seq != result.hosps_personnel) answer = 0;
   if (res_checkin_seq != result.total_hosps_v) answer = 0;
   if (res_village_seq != result.total_in_village) answer = 0;
   if (res_waiting_seq != result.total_waiting) answer = 0;
   if (res_assess_seq != result.total_assess) answer = 0;
   if (res_inside_seq != result.total_inside) answer = 0;

   fprintf(stdout,"\n");
   fprintf(stdout,"Sim. Variables      = expect / result\n");
   fprintf(stdout,"Total population    = %6d / %6d people\n", (int)   res_population_seq, (int) result.total_patients);
   fprintf(stdout,"Hospitals           = %6d / %6d people\n", (int)   res_hospitals_seq, (int) result.hosps_number);
   fprintf(stdout,"Personnel           = %6d / %6d people\n", (int)   res_personnel_seq, (int) result.hosps_personnel);
   fprintf(stdout,"Check-in's          = %6d / %6d people\n", (int)   res_checkin_seq, (int) result.total_hosps_v);
   fprintf(stdout,"In Villages         = %6d / %6d people\n", (int)   res_village_seq, (int) result.total_in_village);
   fprintf(stdout,"In Waiting List     = %6d / %6d people\n", (int)   res_waiting_seq, (int) result.total_waiting);
   fprintf(stdout,"In Assess           = %6d / %6d people\n", (int)   res_assess_seq, (int) result.total_assess);
   fprintf(stdout,"Inside Hospital     = %6d / %6d people\n", (int)   res_inside_seq, (int) result.total_inside);
   fprintf(stdout,"Average Stay        = %6f / %6f u/time\n", (float) res_avg_stay_seq,(float) result.total_time/result.total_patients);

   my_print(top);

   return answer;
}
/**********************************************************************/
void sim_village_main_seq(struct Village *top)
{
   long i;
   for (i = 0; i < aba; i++) sim_village_seq(top);
}


void sim_village_main_par(struct Village *top)
{
   long i;

   if (if_cutoff) {
      #pragma omp parallel
      #pragma omp single
      #pragma omp task untied
         for (i = 0; i < aba; i++) sim_village_par_if(top);
   }
   else if (manual_cutoff) {
      #pragma omp parallel
      #pragma omp single
      #pragma omp task untied
         for (i = 0; i < aba; i++) sim_village_par_manual(top);
   }
  else {
      #pragma omp parallel
      #pragma omp single
      #pragma omp task untied
         for (i = 0; i < aba; i++) sim_village_par(top);
   }
}

void print_usage() {

   fprintf(stderr, "\n");
   fprintf(stderr, "Usage: %s -[options]\n", "Health");
   fprintf(stderr, "\n");
   fprintf(stderr, "Where options are:\n");
   fprintf(stderr, "  -f <file>  : Health input file (mandatory)\n");
   fprintf(stderr, "  -a <flag> : Set if-cutoff on\n");
   fprintf(stderr, "  -b <flag> : Set manual-cutoff on (choose one or none)\n");
   fprintf(stderr, "  -h         : Print program's usage (this help).\n");
   fprintf(stderr, "\n");

}

int main(int argc, char* argv[]) {

char filename[100];
struct Village *top, *top_seq;
int i;
for (i=1; i<argc; i++) {
     if (argv[i][0] == '-') {
       switch (argv[i][1]) {
           case 'f': /* read argument size 0 */
                  argv[i][1] = '*';
                  i++;
                  if (argc == i) { "Erro\n"; exit(100); }
                  strcpy(filename, argv[i]);
                  break;
           case 'a': /* read argument size 0 */
                  argv[i][1] = '*';
                  //i++;
                  //if (argc == i) { "Erro\n"; exit(100); }
                  if_cutoff = 1;
                  manual_cutoff = 0;
                  break;
           case 'b': /* read argument size 0 */
                  argv[i][1] = '*';
                  //i++;
                  //if (argc == i) { "Erro\n"; exit(100); }
                  manual_cutoff = 1;
                  if_cutoff = 0;
                  break;
            case 'h': /* print usage */
                  argv[i][1] = '*';
                  print_usage();
                  exit (100);
                  break;
          }
     }
}


//strcpy(filename,"../input/small.input");
read_input_data_par(filename);

allocate_village(&top, NULL, NULL, sim_level, 0);


double t_start, t_end;

t_start = rtclock();
sim_village_main_par(top);
t_end = rtclock();
fprintf(stdout, "\nParallel Runtime: %0.6lfs\n", t_end - t_start);

int ans_par = check_village_par(top);

read_input_data_seq(filename);
allocate_village(&top_seq, NULL, NULL, sim_level, 0);

t_start = rtclock();
sim_village_main_seq(top_seq);
t_end = rtclock();
fprintf(stdout, "\nSequential Runtime: %0.6lfs\n", t_end - t_start);

int ans_seq = check_village_seq(top_seq);


if (ans_par && ans_seq) {
   fprintf(stdout, "Result: Successful\n");
} else {
   fprintf(stdout, "Result: Unsuccessful\n");
}

return 0;
}

