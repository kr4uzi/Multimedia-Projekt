#pragma once
#include <string>
#include <vector>
#include <cmath>	// fabs
#include <utility>	// move, pair
#include <cassert>

namespace svm
{
	class sparse_vector
	{
	public:
		typedef long size_type;
		typedef float value_type;

		class const_iterator
		{
			friend class sparse_vector;

		private:
			void * _ptr;

		private:
			const_iterator(void * ptr) : _ptr(ptr) { }

		public:
			size_type index() const;

			bool operator==(const const_iterator& rhs) const { return _ptr == rhs._ptr; }
			bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
			void operator++();
			value_type operator*() const;
		};

	private:
		long _size;
		void * _svector;
		void * _words_end;

	private:
		typedef std::pair<size_type, value_type> word;
		static void svector_init(void ** svector, const std::vector<word>& words, void ** words_end);

	public:
		//sparse_vector(const sparse_vector& rhs);
		sparse_vector(sparse_vector&& rhs);
		sparse_vector& operator=(sparse_vector&& rhs);

		template<class T>
		sparse_vector(T begin, T end, size_type size)
			: _size(size)
		{
			std::vector<word> words;
			words.reserve(size);
			size_type i = 1;
			while (begin != end)
			{
				if (std::fabs(*begin) > 0)
					words.emplace_back(i, *begin);

				++begin;
				++i;
			}

			svector_init(&_svector, words, &_words_end);
		}

		~sparse_vector();

		const void * c_ptr() const { return _svector; }
		size_type size() const { return _size; }

		const_iterator begin() const;
		const_iterator end() const { return const_iterator(_words_end); }
	};

	std::string to_string(const sparse_vector& svec);

	class linear_model
	{
	private:
		void * _model;
		sparse_vector::size_type vec_size;
		double * _linear_weights;
		double _b;

	private:
		static void model_init(void ** model, double ** linear_weights, double * b, std::vector<void *>& docs, std::vector<double>& targets, sparse_vector::size_type vec_size, double c);
		// creates a DOC for a given sparse_vector (which has to be owned on the DOCs livespan)
		void * create_doc(sparse_vector::size_type index, const sparse_vector& svec, double costfactor = 1) const;

	public:
		linear_model(const std::string& filename);

		template<class negatives_type, class positives_type>
		linear_model(const positives_type& positives, const negatives_type& negatives, sparse_vector::size_type vec_size, double c = 1)
			: vec_size(vec_size)
		{

			std::vector<void *> docs;
			std::vector<double> targets;
			docs.reserve(negatives.size() + positives.size());
			targets.insert(targets.end(), positives.size(), +1);
			targets.insert(targets.end(), negatives.size(), -1);

			for (auto& positive : positives)
			{
				assert(positive.size() == vec_size);
				docs.push_back(create_doc((sparse_vector::size_type)docs.size(), positive));
			}

			for (auto& negative : negatives)
			{
				assert(negative.size() == vec_size);
				docs.push_back(create_doc((sparse_vector::size_type)docs.size(), negative));
			}

			model_init(&_model, &_linear_weights, &_b, docs, targets, vec_size, c);
		}

		~linear_model();

		sparse_vector::size_type get_vec_size() const { return vec_size; }

		double classify(const sparse_vector& vec) const;
		
		template<class T>
		double classify(T begin, T end) const
		{
			double sum = 0;
			sparse_vector::size_type i = 1;
			while (begin != end)
			{
				assert(i <= vec_size && "Invalid vec size!");
				sum += _linear_weights[i] * (*begin);
				++begin;
				++i;
			}

			return sum - _b;
		}

		void save(const std::string& filename) const;
	};
}


