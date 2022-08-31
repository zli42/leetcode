# Top Interview Questions Medium Collection

## Array and Strings

### [3Sum](https://leetcode.cn/problems/3sum/)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        if n < 3:
            return []
        
        nums.sort()
        
        res = []
        for i in range(n):
            if nums[i] > 0:
                return res
            
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            l = i + 1
            r = n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif s < 0:
                    l += 1
                else:
                    r -= 1
        return res
```

* 时间复杂度：$O(N^2)$，其中 $N$ 是数组 $nums$ 的长度。
* 空间复杂度：$O(logN)$。我们忽略存储答案的空间，额外的排序的空间复杂度为 $O(logN)$。然而我们修改了输入的数组 $nums$，在实际情况下不一定允许，因此也可以看成使用了一个额外的数组存储了 $nums$ 的副本并进行排序，空间复杂度为 $O(N)$。

### [Set Matrix Zeroes](https://leetcode.cn/problems/set-matrix-zeroes/)

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        first_col = False
        first_row = False
        
        for i in range(m):
            if matrix[i][0] == 0:
                first_col = True
                break
                
        for j in range(n):
            if matrix[0][j] == 0:
                first_row = True
                break
                
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                    
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
                    
        if first_col:
            for i in range(m):
                matrix[i][0] = 0
            
        if first_row:
            for j in range(n):
                matrix[0][j] = 0
```

* 时间复杂度：$O(mn)$，其中 $m$ 是矩阵的行数，$n$ 是矩阵的列数。我们至多只需要遍历该矩阵两次。
* 空间复杂度：$O(1)$。我们只需要常数空间存储若干变量。






## Tree and Graphs

## Sorting and Searching

## Dynamic Programming