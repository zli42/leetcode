### [Jump Game](https://leetcode.cn/problems/jump-game/)

python
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        max_len = 0
        
        for i in range(n):
            if i <= max_len:
                max_len = max(max_len, i + nums[i])
                if  max_len >= n - 1:
                    return True
                
        return False
```

rust
```rust
impl Solution {
    pub fn can_jump(nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut right_most = 0;
        for i in 0..n {
            if i > right_most {
                return false;
            }

            right_most = std::cmp::max(right_most, i + nums[i] as usize);

            if right_most >= n - 1 {
                return true;
            }
        }

        false
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的大小。只需要访问 `nums` 数组一遍，共 `n` 个位置。
* 空间复杂度：$O(1)$，不需要额外的空间开销。

### [ Coin Change](https://leetcode.cn/problems/coin-change/)

python
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf') for _ in range(amount + 1)]
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
        return dp[amount] if dp[amount] != float('inf') else -1
```

rust
```rust
impl Solution {
    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let mut dp = vec![amount + 1; amount as usize + 1];
        dp[0] = 0;
        for i in 1..=amount {
            for coin in coins.iter() {
                if i - coin < 0 {
                    continue;
                }
                dp[i as usize] = std::cmp::min(dp[i as usize], dp[(i - coin) as usize] + 1)
            }
        }
        if dp[amount as usize] == amount + 1 {
            -1
        } else {
            dp[amount as usize]
        }
    }
}
```

* 时间复杂度：$O(Sn)$，其中 $S$ 是金额，$n$ 是面额数。我们一共需要计算 $O(S)$ 个状态，$S$ 为题目所给的总金额。对于每个状态，每次需要枚举 $n$ 个面额来转移状态，所以一共需要 $O(Sn)$ 的时间复杂度。
* 空间复杂度：$O(S)$。数组 `dp` 需要开长度为总金额 $S$ 的空间。

### [Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

python
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        d = []
        for num in nums:
            if not d or num > d[-1]:
                d.append(num)
            else:
                l = 0
                r = len(d) - 1
                loc = r
                while l <= r:
                    mid = (l + r) // 2
                    if d[mid] >= num:
                        loc = mid
                        r = mid - 1
                    else:
                        l = mid + 1
                d[loc]  = num
                
        return len(d)
```

rust
```rust
impl Solution {
    pub fn length_of_lis(nums: Vec<i32>) -> i32 {
        let mut list = vec![&nums[0]];  // list 并非实际的最长递增子序列，但是长度是对的
        for num in nums.iter().skip(1) {
            let n = list.len();
            if num > list[n - 1] {
                list.push(num);
            } else {
                let mut left = 0;
                let mut right = n;
                while left < right {
                    let mid = left + (right - left) / 2;
                    if num > list[mid] {
                        left = mid + 1;
                    } else {
                        right = mid
                    }
                }
                list[left] = num;
            }
        }
        list.len() as i32
    }
}
```

* 时间复杂度：$O(n\log n)$。数组 `nums` 的长度为 `n`，我们依次用数组中的元素去更新 `d` 数组，而更新 `d` 数组时需要进行 $O(\log n)$ 的二分搜索，所以总时间复杂度为 $O(n\log n)$。
* 空间复杂度：$O(n)$，需要额外使用长度为 $n$ 的 `d` 数组。
