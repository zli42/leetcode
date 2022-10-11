### [Climbing Stairs](https://leetcode.cn/problems/climbing-stairs/)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        p = 0
        q = 0
        r = 1
        for _ in range(n):
            p = q
            q = r
            r = p + q
        return r
```

* 时间复杂度：循环执行 $n$ 次，每次花费常数的时间代价，故渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里只用了常数个变量作为辅助空间，故渐进空间复杂度为 $O(1)$。

### [Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

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

```rust
impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut maxprofit = 0;
        let mut minprice = prices[0];
        for price in prices {
            maxprofit = std::cmp::max(price - minprice, maxprofit);
            minprice = std::cmp::min(price, minprice);
        }
        maxprofit
    }
}
```

* 时间复杂度：$O(n)$，只需要遍历一次。
* 空间复杂度：$O(1)$，只使用了常数个变量。

### [Maximum Subarray](https://leetcode.cn/problems/maximum-subarray/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre = 0
        res = nums[0]
        for num in nums:
            pre = max(num, pre + num)
            res = max(res, pre)
        return res
```

```rust
impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut pre = 0;
        let mut res = nums[0];
        for num in nums {
            pre = std::cmp::max(pre + num, num);
            if num > 0 
            res = std::cmp::max(res, pre);
        }
        res
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为 `nums` 数组的长度。我们只需要遍历一遍数组即可求得答案。
* 空间复杂度：$O(1)$。我们只需要常数空间存放若干变量。

### [House Robber](https://leetcode.cn/problems/house-robber/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0

        size = len(nums)
        if size == 1:
            return nums[0]
        
        first, second = nums[0], max(nums[0], nums[1])
        for i in range(2, size):
            first, second = second, max(first + nums[i], second)
        
        return second
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组长度。只需要对数组遍历一次。
* 空间复杂度：$O(1)$。使用滚动数组，可以只存储前两间房屋的最高总金额，而不需要存储整个数组的结果，因此空间复杂度是 $O(1)$。
