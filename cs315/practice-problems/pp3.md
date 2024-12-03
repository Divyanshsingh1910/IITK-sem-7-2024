Practice Problems 3

The reference texts contain plenty of exercises for technical topics such as query processing, query optimization, ER diagrams and functional dependence topics.

1. Does the equality $$E_1\bowtie_\theta(E_2 - E_3) = E_1\bowtie_\theta E_2 - E_1\bowtie_\theta&nbsp; E_3$$ always hold? Either justify or give a counter example.

2. Does $$\Pi_A(r - s) = \Pi_A(r) - \Pi_A(s)$$ always hold? Either justify or give a counter example.

3. Does $$A ⟖ (B ⟖ C ) = (A ⟖ B) ⟖ C $$ always hold? Either justify or give a counter example. Here ⟖ is the right outer join.

4. Under what conditions on the predicate $$\theta$$ will the following (otherwise false) equality hold? $$\sigma_\theta(A ⟕ B) = \sigma_\theta(A \bowtie B)$$. Here ⟕ is the left outer join.

5. Write an SQL query to print "YES" if a given column contains exactly 5 unique values and "NO" otherwise.

6. Consider a relation $$R(A,B,C,D,E,F)$$ and a set of FDs namely $$\mathcal F=\{AB\rightarrow C, BC \rightarrow AD, D \rightarrow&nbsp; E, CD \rightarrow B\}$$. Does the functional dependency $$D \rightarrow A$$ follow from $$\mathcal F$$? Justify using Armstrong axioms and the attribute set closure algorithm.

7. Consider a relation $$R(A,B,C,D,E)$$ and a set of FDs namely $$\mathcal F=\{A\rightarrow BC, CD \rightarrow E, B \rightarrow&nbsp; D, E \rightarrow A\}$$. Find all candidate keys of this relation.

8. Given a schema $$R$$ and subset schemata $$R_1,R_2,\ldots,R_K \subseteq R$$ i.e. all the $$R_k$$ are subsets of columns. If $$R = \bigcup_{k\in [K]}R_k$$ then we call this a decomposition. Show that if $$r$$ is a table/relation with schema $$R$$ then $$r \subseteq \Pi_{R_1}(r) \bowtie \Pi_{R_2}(r) \bowtie \ldots \Pi_{R_K}(r)$$.
