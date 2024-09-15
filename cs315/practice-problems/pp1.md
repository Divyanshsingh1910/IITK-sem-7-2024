# Practice Problem #1

## Sqlite querries on chinook.db

### Questions

1. Customer City Name. Write an SQL query that uses the customers table to count the number of customers who live in a city whose name starts with the letter 'R'.

2. Employee Workload. For every employee in the employees table, find out the number of customers served by that employee. The number must be reported for every employee, even if it is zero.

3. Employees of the Month. Using the results from the above task, you can find the average number of customers served by an employee. Find all employees who are serving more than average number of customers.

4. Sum-of-squares: Write down an SQL query to find the sum of the squares of values in the total column in the invoices table.

5. Implementing variance in SQLite: SQLite supports several aggregation operators such as COUNT, SUM, MAX etc. However, unlike MySQL that also provides a standard deviation operator STDEV, SQLite does not provide one. Write down an SQLite query to find the variance of the total column in the invoices table. Recall that the variance of $n$ real numbers $x_1, \dots, x_n$ is defined to be $\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$ where $\mu = \frac{1}{n} \sum_{i=1}^n x_i$. The standard deviation is defined to be the square root of the variance.

6. Genre Search. Using the genres and tracks tables, find all genres where the genre name starts with 'R' and there are at least 10 tracks of that genre.

7. Advanced Genre Search. Using the genres and tracks tables, find all genres where the genre name starts with 'R' and the number of tracks of that genre at least 1.5 times the average number of tracks per genre.
