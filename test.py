def max_subset_sum_less_than_or_equal_to_Y(X, Y):
    n = len(X)
    # Create a table to store the maximum subset sum for each possible sum up to Y
    dp = [0] * (Y + 1)

    for i in range(n):
        for j in range(Y, 0, -1):
            if X[i] <= j:
                dp[j] = max(dp[j], dp[j - X[i]] + X[i])
        print(dp)
    return dp[Y]

# Example usage:
X = [2, 3, 7, 11]
Y = 30
result = max_subset_sum_less_than_or_equal_to_Y(X, Y)
print("Maximum subset sum less than or equal to", Y, "is:", result)