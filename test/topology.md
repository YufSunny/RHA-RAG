# Topological Spaces

**Definition 2.1 (Topological Space).** A topological space is a pair (X, tau)
where X is a set and tau is a collection of subsets of X satisfying:
1. The empty set and X belong to tau.
2. The union of any collection of sets in tau is in tau.
3. The intersection of any finite collection of sets in tau is in tau.

**Definition 2.2 (Open Set).** The members of tau are called open sets.

**Definition 2.3 (Closed Set).** A subset C of X is closed if its complement
X \ C is open.

**Theorem 2.4 (Properties of Closed Sets).** In any topological space:
(a) The empty set and X are closed.
(b) Arbitrary intersections of closed sets are closed.
(c) Finite unions of closed sets are closed.

**Definition 2.5 (Continuous Function).** A function f: X -> Y between
topological spaces is continuous if the preimage of every open set in Y
is open in X.

**Definition 2.6 (Compactness).** A space X is compact if every open cover
has a finite subcover.

**Theorem 2.7 (Heine-Borel).** A subset of R^n is compact iff it is closed
and bounded.

**Definition 2.8 (Connectedness).** A space X is connected if it cannot be
written as the union of two disjoint nonempty open sets.
