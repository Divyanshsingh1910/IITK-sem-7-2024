Practice Problems 2

Please note that questions on "theoretical topics" such as B+trees, external sorting etc are not expected to fixate exact implementation details. Instead they may examine general understanding of concepts.

1. The symmetric difference of two sets $$A \Delta B$$ is defined to be the set of all elements that are present in exactly one of the two sets (i.e., not in both). How would you implement the $$\Delta$$ operator in Relational Algebra? How would you implement it as an SQLite query?

2. MySQL has a operator called SOME that performs comparison against a list and returns TRUE if even one of the comparisons is TRUE. This allows us to write queries such as the following which finds all students who have a roll no greater than some EE student.

```
SELECT RollNo FROM Student
WHERE RollNo > SOME (
    SELECT RollNo FROM Student
    WHERE DEPT = 'EE'
)
```

Implement the above query in SQLite that does not have the SOME operator.

3. Given the following SQL query
`SELECT __(A)__ FROM Student WHERE RollNo = __(B)__`
where (A) and (B) are user inputs taken from a web-based form, perform an injection attack to instead print the roll number of the highest CPI student instead. Note that (A) and (B) must both be non-null but may contain arbitrary characters. Assume that a semi-colon will terminate a query and start a new one.

- Consider a B+ tree on table Student constructed on the primary key RollNo where each non-leaf node has exactly 4 children, all leaves are at depth 5 (root being depth 0). The leaves are all strung together using sibling pointers. Assume that a node can be stored in a single disk block and the nodes of the tree are stored contiguously in a breadth first fashion (nodes in a given level stored from left to right).

1. Find the number of leaf nodes in the tree
2. Find the total number of nodes in the tree
3. Find the number of disk seeks and disk block&nbsp; reads will it take to locate a record with a given RollNo.
4. Suppose we wish to retrieve all records with RollNo in the range $$[R1,R2]$$. It is known that $$K$$ records satisfy the condition. It is known that $$B$$ records fit inside a block. How many disk seeks and disk block&nbsp; reads will it take to output all records?
