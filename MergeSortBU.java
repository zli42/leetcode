import java.util.Arrays;
import java.lang.Math;

public class MergeSortBU{
	private static int[] temp;
	
	public static void sort(int[] a){
		int n = a.length;
		temp = new int[n];
		for(int sz = 1; sz < n; sz = sz+sz){
			for(int lo = 0; lo < n-sz; lo += sz+sz){
				int mid = lo + sz - 1;
				int hi = Math.min(lo+sz+sz-1, n-1);
				merge(a, lo, mid, hi);				
			}
		}
	}
	
	private static void merge(int[] a, int lo, int mid, int hi){
		for(int k = lo; k <= hi; k++){
			temp[k] = a[k];
		}
		int i = lo, j = mid + 1;
		for(int k = lo; k <= hi; k++){
			if(i > mid){
				a[k] = temp[j++];
			}else if(j > hi){
				a[k] = temp[i++];
			}else if(temp[j] < temp[i]){
				a[k] = temp[j++];
			}else{
				a[k] = temp[i++];
			}
		}
	}
	
	public static void main(String[] args){
		int[] a = {2, 5, 8, 6, 4, 9, 0, 6, 7, 1, 3};
        MergeSortBU.sort(a);
		System.out.println(Arrays.toString(a));
	}
}