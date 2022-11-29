# Dynamic Programming [easy]

## [Climbing Stairs](https://leetcode.cn/problems/climbing-stairs/)

python

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return 1

        dp = [0 for _ in range(n)]
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n-1]

class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n

        first = 1
        second = 2
        for _ in range(3, n+1):
            cur = first + second
            first = second
            second = cur
        return second
```

rust

```rust
impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        if n <= 2 {
            return n;
        }

        let mut first = 1;
        let mut second = 2;
        for _ in 3..=n {
            let cur = first + second;
            first = second;
            second = cur;
        }
        second
    }
}
```

* 时间复杂度：循环执行 $n$ 次，每次花费常数的时间代价，故渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里只用了常数个变量作为辅助空间，故渐进空间复杂度为 $O(1)$。

## [Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

python

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxprofit = 0
        minprice = prices[0]
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)
            minprice = min(price, minprice)
        return maxprofit
```

rust

```rust
impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut maxprofit = 0;
        let mut min_price = prices[0];
        for price in prices.iter() {
            maxprofit = std::cmp::max(maxprofit, price - min_price);
            min_price = std::cmp::min(min_price, price);
        }
        maxprofit
    }
}
```

* 时间复杂度：$O(n)$，只需要遍历一次。
* 空间复杂度：$O(1)$，只使用了常数个变量。

## [Maximum Subarray](https://leetcode.cn/problems/maximum-subarray/)

python

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i - 1] > 0 :
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre = 0
        res = nums[0]
        for num in nums:
            pre = max(num, pre + num)
            res = max(res, pre)
        return res
```

rust

```rust
impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut pre = 0;
        let mut res = nums[0];
        for num in nums.iter() {
            pre = std::cmp::max(pre + num, num);
            res = std::cmp::max(res, pre);
        }
        res
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为 `nums` 数组的长度。我们只需要遍历一遍数组即可求得答案。
* 空间复杂度：$O(1)$。我们只需要常数空间存放若干变量。

## [House Robber](https://leetcode.cn/problems/house-robber/)

python

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[n - 1]

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        first = nums[0]
        second = max(nums[0], nums[1])
        for i in range(2, n):
            cur = max(first + nums[i], second)
            first = second
            second = cur
        return second
```

rust

```rust
impl Solution {
    pub fn rob(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 {
            return nums[0];
        }

        let mut first = nums[0];
        let mut second = std::cmp::max(nums[0], nums[1]);
        for i in 2..n {
            let cur = std::cmp::max(first + nums[i], second);
            first = second;
            second = cur;
        }
        second
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组长度。只需要对数组遍历一次。
* 空间复杂度：$O(1)$。使用滚动数组，可以只存储前两间房屋的最高总金额，而不需要存储整个数组的结果，因此空间复杂度是 $O(1)$。
