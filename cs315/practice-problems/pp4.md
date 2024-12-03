Practice Problems

Q1. Assume&nbsp; a relation R with 1000 tuples and two attributes/columns named A and B. There are 5 distinct values for attribute A and 12 distinct values forattribute B. Find the estimated size of the query $$\sigma_{(A=a \vee B=b)}(R)$$.

Q2. Given a table with 3 columns $$A, B, C$$, write an SQL query to output YES if the table satisfies the dependency $$AB\rightarrow C$$ else output NO.

Q3. Out of the two concurrency protocols, namely two-phase locking and timestamp-ordering, which protocol(s) always ensures conflict serializability? Which protocol(s) ensures freedom from deadlock? It may be that a specific version of these protocols ensures these properties.

Q4. Consider the two transactions
$$T_1$$
read(A)
read(B)
if A = 0, B := B+1
write(B)
$$T_2$$
read(B)
read(A)
if B = 0, A := A+1
write(A)

The consistency requirement specified by the client application is $$A\cdot B = 0$$ i.e. the product must be zero. Initially both A, B are set to 0.
Will every serial execution of these two transactions i.e. $$T_1$$ followed by $$T_2$$ or vice versa, always preserve consistency?Give an example of a concurrent/interleaved execution of the two transactions that is not conflict serializable.Give an example of a concurrent/interleaved execution of the two transactions that is conflict serializable.Add lock and unlock instructions to the two transactions so as to obey two-phase locking. Can an interleaved execution with such locks result in a deadlock?

Q5. Let R(A, B, C, D, E) be decomposed into relations with the following three sets of attributes: {A, B, C}, {B, C, D}, and {A, C, E}. For eachof the following sets of FD’s, use the chase test to tell whether the decomposition is lossless.
B → E and CE → A.AC → E and BC → DA → D , D → E , and B → D.A → D , CD → E , and E → D

