// #ifndef __CHORD_H__
// #define __CHORD_H__

// CONTOURTILER_BEGIN_NAMESPACE

// class Chord
// {
// public:
//   friend std::size_t hash_value(const Chord& chord)
//   {
//     throw std::logic_error("Just making sure!");
//     std::size_t seed = 0;
//     boost::hash_combine(seed, chord._vertices[0]);
//     boost::hash_combine(seed, chord._vertices[1]);
//     return seed;
//   }

//   bool operator==(const Chord& chord)
//   {
//     return _vertices[0] == chord._vertices[0] && _vertices[1] ==
//     chord._vertices[1];
//   }

// public:
//   Chord() {}
//   Chord(const Contour_vertex& v0, const Contour_vertex& v1)
//   {
//     if (v0.contour() == v1.contour())
//       throw new std::logic_error("The two vertices on a tile must lie on
//       different contours");
//     // This restriction is so that the hashing function is consistent
//     if (v0.contour()->slice() > v1.contour()->slice())
//       throw new std::logic_error("The first vertex must be on a lower slice
//       than the second vertex");
//     _vertices[0] = v0; _vertices[1] = v1;
//   }
//   ~Chord() {}

// private:
//   Contour_vertex _vertices[2];
// };

// CONTOURTILER_END_NAMESPACE

// #endif
