#pragma once

#ifndef _SVM_LIGHT_SPARSE_VECTOR
#define _SVM_LIGHT_SPARSE_VECTOR

//If the #define is not commented out, we use no element proxies, i.e. entries
//in a vector/matrix can be accessed directly (e.g. by vec[ii] for the ii-th
//entry in vec). However, if the #define is commented out we have to 
//access the respective element by vec[ii].ref()
//see BoostUblasTest.h for some coding examples, also see
//http://www.boost.org/doc/libs/1_41_0/libs/numeric/ublas/doc/operations_overview.htm
#define BOOST_UBLAS_NO_ELEMENT_PROXIES

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//#include <boost/numeric/ublas/operation.hpp>
//#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::compressed_vector<double> SparseVector;

#endif
