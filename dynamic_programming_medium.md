# Dynamic Programming [medium]

## [Jump Game](https://leetcode.cn/problems/jump-game/)

python

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        max_idx = 0
        i = 0
        while i <= max_idx:
            max_idx = max(max_idx, i + nums[i])
            if max_idx >= n - 1:
                return True
            i += 1
        return False
```

rust

```rust
impl Solution {
    pub fn can_jump(nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut max_idx = 0;
        let mut i = 0;
        while i <= max_idx {
            max_idx = std::cmp::max(max_idx, i + nums[i] as usize);
            if max_idx >= n - 1 {
                return true;
            }
            i += 1;
        }
        false
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的大小。只需要访问 `nums` 数组一遍，共 `n` 个位置。
* 空间复杂度：$O(1)$，不需要额外的空间开销。

## [Coin Change](https://leetcode.cn/problems/coin-change/)

python

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount+1 for _ in range(amount+1)]
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                rest = i - coin
                if rest < 0:
                    continue
                dp[i] = min(dp[i], dp[rest] + 1)
        return dp[amount] if dp[amount] != amount + 1 else -1
```

rust

```rust
impl Solution {
    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let mut dp = vec![amount + 1; amount as usize + 1];
        dp[0] = 0;
        for i in 1..=amount {
            for &coin in coins.iter() {
                let rest = i - coin;
                if rest < 0 {
                    continue;
                }
                dp[i as usize] = std::cmp::min(dp[i as usize], dp[rest as usize] + 1);
            }
        }
        if dp[amount as usize] == amount + 1 {
            return -1;
        } else {
            return dp[amount as usize];
        }
    }
}
```

* 时间复杂度：$O(Sn)$，其中 $S$ 是金额，$n$ 是面额数。我们一共需要计算 $O(S)$ 个状态，$S$ 为题目所给的总金额。对于每个状态，每次需要枚举 $n$ 个面额来转移状态，所以一共需要 $O(Sn)$ 的时间复杂度。
* 空间复杂度：$O(S)$。数组 `dp` 需要开长度为总金额 $S$ 的空间。

## [Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

把小的尽量往前塞

python

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1 for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = []
        for num in nums:
            if not tails or num > tails[-1]:
                tails.append(num)
            else:
                left = 0
                right = len(tails)
                while left < right:
                    mid = left + (right - left) // 2
                    if tails[mid] < num:
                        left = mid + 1
                    else:
                        right = mid
                tails[left] = num
        return len(tails)
```

rust

```rust
impl Solution {
    pub fn length_of_lis(nums: Vec<i32>) -> i32 {
        let mut tails = vec![nums[0]];
        for num in nums.into_iter().skip(1) {
            let n = tails.len();
            if num > tails[n - 1] {
                tails.push(num);
            } else {
                let mut left = 0;
                let mut right = n;
                while left < right {
                    let mid = left + (right - left) / 2;
                    if tails[mid] < num {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                tails[left] = num;
            }
        }
        tails.len() as i32
    }
}
```

* 时间复杂度：$O(n\log n)$。数组 `nums` 的长度为 `n`，我们依次用数组中的元素去更新 `d` 数组，而更新 `d` 数组时需要进行 $O(\log n)$ 的二分搜索，所以总时间复杂度为 $O(n\log n)$。
* 空间复杂度：$O(n)$，需要额外使用长度为 $n$ 的 `d` 数组。
