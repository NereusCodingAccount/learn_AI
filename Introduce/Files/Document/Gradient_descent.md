# Gradient descent 梯度下降

#### Gradient descent is a method(方法) for unconstrained(無束約的) mathematical optimization(數學最佳化). It is a first-order iterative algorithm(迭代演算法) for minimizing(最小化) a differentiable(可微分化的) multivariate function(多元函數).

### The idea is to take repeated steps(反覆迭代) in the opposite direction of the gradient(梯度的反方向) (or approximate gradient(近似梯度)) of the function(函數) at the current point(當前點), because this is the direction(方向) of steepest descent(最陡). Conversely(相反), stepping in the direction of the gradient(梯度方向) will lead to a trajectory(軌跡) that maximizes(最大化) that function; the procedure(過程) is then known as gradient ascent(梯度上升). It is particularly useful(特別好用) in machine learning(機器學習) for minimizing(最小化) the cost(成本函數) or loss(損失函數) function. Gradient descent should not be confused with **local search algorithms(局部搜尋演算法)**, although both are iterative methods(迭代方法) for optimization(優化).

---

## Description
![alt text](../Pictures/Gd_equation1.png)
#### Gradient descent is based on(基於） the observation that if the multi-variable(多變數） function **_f(x)_** is defined and differentiable(可微分的） in a neighborhood of a point **_a_**, then **_f(x)_** decreases(減少） fastest if one goes from **_a_** in the direction of the negative gradient of **_f_** at **_a_**, - **_Vf(a)_**. It follows that, if
#### for a small enough step size or learning rate **_n € R+_**, then **_f(an) ≥ f(an+1)_**. In other words, the term(術語？） **_nV f(a)_** is subtracted(減去） from **_a_** because we want to move against(反對） the gradient, toward the local minimum. With this observation in mind, one starts with a guess **_x0_** for a local minimum of **_f_**, and considers the sequence(順序） **_x0_**, **_X1_**, **_X2_**, ... such that

![alt text](../Pictures/Gd_equation2.png)
#### We have a monotonic sequence
#### so the sequence (**_Xn_**) converges to the desired local minimum. Note that the value of the step size **_n_** is allowed to change at every iteration.
#### It is possible to guarantee the convergence to a local minimum under certain assumptions on the function **_f_** (for example, **_f_** convex and **_V f_** Lipschitz) and particular choices of n. Those include the sequence

![alt text](../Pictures/Gd_equation3.png)
#### as in the Barzilai-Borwein method,  or a sequence **_n_** satisfying the Wolfe conditions (which can be found by using line search). When the function **_f_** is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.
#### This process is illustrated in the adjacent picture. Here, **_f_** is assumed to be defined on the plane, and that its graph has a bowl shape. The blue curves are the contour lines, that is, the regions on which the value of fis constant. A red arrow originating at a point shows the direction of the negative gradient at that point. Note that the (negative) gradient at a point is orthogonal to the contour line going through that point. We see that gradient descent leads us to the bottom of the bowl, that is, to the point where the value of the function **_f_** is minimal.

![alt text](../Pictures/Gd_p1.png)

### An analogy(類比) for understanding gradient descent
The basic intuition(基本直覺)behind gradient descent can be illustrated(闡述)by a hypothetical scenario(假設場景). People are stuck(卡住) in the mountains and are trying to get down (i.e.(也就是說), trying to find the global minimum(全域最小值)). There is heavy fog(霧) such that visibility(能見度)is extremely(極為) low. Therefore(所以), the path down the mountain is not visible, so they must use local information to find the minimum. They can use the method(方法)of gradient descent, which involves(涉及) looking at the steepness(陡峭度) of the hill at their current position(目前位置), then proceeding(繼續)in the direction(方向)with the steepest(最陡峭)descent (i.e., downhill(下坡)). If they were trying to find the top of the mountain (i.e., the maximum), then they would proceed(繼續) in the direction of steepest ascent(上升) (i.e., uphill(上坡)). Using this method, they would eventually(最終) find their way down the mountain or possibly get stuck in some hole (i.e., local minimum or saddle point(鞍點)), like a mountain lake. However, assume(認為)also that the steepness(陡峭度) of the hill is not **"immediately obvious(顯而易見)"** with simple observation, but rather(相當)it requires(要求) a sophisticated(複雜的)instrument(工具、機械、儀器)to measure, which the people happen to have at that moment(片刻). It takes quite some time to measure the steepness of the hill with the instrument. Thus(因此), they should minimize(最小化) their use of the instrument if they want to get down the mountain before sunset. The difficulty then is choosing the frequency(頻率) at which they should measure the steepness of the hill so as not to go off track(追蹤).

In this analogy(類比), the people represent the algorithm(演算法), and the path taken down(紀錄)the mountain represents(代表)the sequence(順序)of parameter(範圍)settings that the algorithm will explore(探索). The steepness of the hill represents the slope(坡)of the function at that point. The instrument used to measure steepness is differentiation(差異化). The direction(方向)they choose to travel in aligns(對齊) with the gradient of the function at that point. The amount(數量) of time they travel before taking another measurement(測量) is the step size(步長).

### Choosing the step size and descent direction
![alt text](../Pictures/Gd_equation4.png)
#### Since using a step size n that is too small would slow convergence, and a n too large would lead to overshoot and divergence, finding a good setting of n is an important practical problem. Philip Wolfe also advocated using "clever choices of the [descent] direction" in practice (10) While using a direction that deviates from the steepest descent direction may seem counter-intuitive, the idea is that the smaller slope may be compensated for by being sustained over a much longer distance.
#### To reason about this mathematically, consider a direction Pn and step size in and consider the more general update:
#### Finding good settings of Pn and n requires some thought. First of all, we would like the update direction to point downhill. Mathematically, letting On denote the angle between - Vf(an) and Pn, this requires that cos On > 0. To say more, we need more information about the objective function that we are optimising. Under the fairly weak assumption that fis continuously differentiable, we may prove that:

![alt text](../Pictures/Gd_equation5.png)
#### This inequality implies that the amount by which we can be sure the function f is decreased depends on a trade off between the two terms in square brackets. The first term in square brackets measures the angle between the descent direction and the negative gradient. The second term measures how quickly the gradient changes along the descent direction.
#### In principle inequality (1) could be optimized over P, and n to choose an optimal step size and direction. The problem is that evaluating the second term in square brackets requires evaluating Vf(an - thnPn), and extra gradient evaluations are generally expensive and undesirable. Some ways around this problem are:
#### - Forgo the benefits of a clever descent direction by setting P, = V f(an), and use line search to find a suitable step-size Yn, such as one that satisfies the Wolfe conditions. A more economic way of choosing learning rates is backtracking line search, a method that has both good theoretical guarantees and experimental results. Note that one does not need to choose Pn to be the gradient; any direction that has positive inner product with the gradient will result in a reduction of the function value (for a sufficiently small value of **_nn_**).

![alt text](../Pictures/Gd_equation6.png)
#### - Assuming that f is twice-differentiable, use its Hessian V2 f to estimate
#### Then choose Pn and in by optimising
inequality (1).
#### - Assuming that V f is Lipschitz, use its Lipschitz constant L to bound
#### Then choose Pn and An by optimising inequality
（1）.
#### Build a custom model of {        } for f. Then choose Pn and An
by optimising inequality (1).
#### - Under stronger assumptions on the function f such as convexity, more advanced techniques may be possible.
#### Usually by following one of the recipes above, convergence to a local minimum can be guaranteed.
When the function fis convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.

---

## Solution of a linear system(線型函數的解)
Gradient descent can be used to solve a **system of linear equations**
![alt text](../Pictures/Gd_S_ls1.png)

reformulated(重新配方) as a quadratic(二次函數) minimization(最小化) problem. If the system matrix **_A_** is real symmetric(對稱) and positive-definite(正定矩陣), an objective function is defined(定義) as the quadratic function, with minimization of
![alt text](../Pictures/Gd_S_ls2.png)

so that
![alt text](../Pictures/Gd_S_ls3.png)

For a general real matrix(矩陣) **_A_**, linear(線型) least(最小)squares(平方法) define(定義)
![alt text](../Pictures/Gd_S_ls4.png)

In traditional linear least squares for real **_A_** and **_b_** the **Euclidean norm(歐幾里德範數)is used, in which case
![alt text](../Pictures/Gd_S_ls5.png)

![alt text](../Pictures/Gd_S_ls6.png)

![alt text](../Pictures/Gd_S_ls7.png)

### Geometric(幾何) behavior(行為) and residual(殘差) orthogonality(正交性)
![alt text](../Pictures/Gd_S_ls8.png)
As shown in the image on the right, steepest descent converges(收斂) slowly due to the high condition number of**_A_**, and the orthogonality(正交性) of residuals(殘差) forces(力量) each new direction to undo(撤銷) the overshoot(過衝) from the previous(以前的) step. The result is a path that zigzags(蜿蜒) toward(朝向) the solution. This inefficiency(效率低下) is one reason **conjugate gradient(共軛梯度)** or preconditioning methods(預處理方法) are preferred(首選).

---

## Solution of a non-linear system(非線型系統)
Gradient descent can also be used to solve a system of nonlinear equations. Below(以下) is an example that shows how to use the gradient descent to solve for three unknown variables(未知變數), **_x1_**, **_x2_**, and **_x3_**. This example shows one iteration(迭代) of the gradient descent.

![alt text](../Pictures/Gd_nl_S1.png)
![alt text](../Pictures/Gd_nl_S2.png)
![alt text](../Pictures/Gd_nl_S3.png)
![alt text](../Pictures/Gd_nl_S4.png)

---

## Comments

#### Gradient descent works in spaces of any number of dimensions, even in infinite-dimensional(無限維度) ones. In the latter case(後者狀況), the search space is typically a function space, and one calculates the Fréchet derivative(衍生物) of the functional(功能性) to be minimized to determine(決定) the descent direction(下降方向).

#### That gradient descent works in any number of dimensions(方面) (finite(有限個) number at least) can be seen as a consequence(結果) of the Cauchy–Schwarz inequality(不等式), i.e.(=拉丁文id est 也就是說) the magnitude(規模) of the inner(內在的) (dot) product of two vectors(向量) of any dimension(方面) is maximized(最大化) when they are colinear(共線). In the case of gradient descent, that would be when the vector of independent(獨立的) variable adjustments(調整) is proportional(比例) to the gradient vector(向量) of partial(部分) derivatives(衍生物).

#### The gradient descent can take many iterations(迭代) to compute(計算) a local minimum with a required(所需的) accuracy(準確性), if the curvature(曲率) in different directions is very different for the given(給定的) function. For such functions, preconditioning(預處理), which changes the geometry(幾何學) of the space to shape the function level sets like concentric(同心) circles, cures the slow convergence(收斂). Constructing(建造) and applying(申請) preconditioning(預處理) can be computationally(計算地) expensive(成本昂貴), however.

#### The gradient descent can be modified via momentums(動量) (Nesterov, Polyak, and Frank–Wolfe) and heavy-ball parameters(參數) (exponential(指數) moving averages and positive-negative(正負) momentum(動量)). The main examples of such optimizers(最佳化器) are Adam, DiffGrad, Yogi, AdaBelief, etc(等等).

![alt text](../Pictures/Gd_c.png)

---

## Modifications(修改)

#### Gradient descent can converge(收斂) to a local minimum and slow down in a neighborhood(鄰近) of a saddle point(鞍點). Even for unconstrained(不可約束的) quadratic(二次的) minimization(縮減到最小), gradient descent develops(發展) a zig–zag(曲折的) pattern(模式) of subsequent(隨後的) iterates(迭代) as iterations(迭代) progress(進度), resulting in slow convergence(收斂). Multiple modifications(多次修改) of gradient descent have been proposed(提議) to address(解決) these deficiencies(不足).

### Fast gradient methods

![alt text](../Pictures/Gd_M.png)

### Momentum(動力)  or heavy ball method

#### Trying to break the zig-zag(曲折的) pattern(圖案) of gradient descent, the momentum or heavy ball method uses a momentum term(術語) in analogy(類比) to a heavy ball sliding(滑) on the surface(表面) of values of the function being minimized, or to mass(大量的) movement in Newtonian dynamics(動力學) through a viscous(稠密的) medium(介質) in a conservative force field(保守力場). Gradient descent with momentum(動量) remembers(記住) the solution update at each iteration(迭代), and determines the next update as a linear combination(組合) of the gradient and the previous(以前的) update. For unconstrained(不受約束的) quadratic(二次函數) minimization(最小化), a theoretical(理論) convergence(收斂) rate(速率) bound(邊界) of the heavy ball method is asymptotically(漸近的) the same as that for the optimal(最佳的) conjugate(共軛) gradient method.

#### This technique(科技) is used in stochastic(隨機) gradient descent and as an extension(擴大) to the backpropagation(反向傳播) algorithms(演算法) used to train artificial neural(神經) networks. In the direction of updating, stochastic(隨機) gradient descent adds a stochastic(隨機) property(財產). The weights can be used to calculate the derivatives(衍生物).
---

## Extensions

#### Gradient descent can be extended(延長） to handle(處理） constraints(約束） by including(包含） a projection(投射） onto the set of constraints. This method is only feasible(可行的） when the projection is efficiently(高效率） computable(可計算的） on a computer. Under suitable(合適的） assumptions(假設）, this method converges(收斂）. This method is a specific case of the forward–backward(前進後退） algorithm(演算法） for monotone(單調） inclusions(包容性） (which includes convex凸面） programming and variational inequalities(變分不等式）).

#### Gradient descent is a special case of mirror descent using the squared Euclidean distance as the given Bregman divergence.

---

## Theoretical properties(理論性質）
#### The properties(財產） of gradient descent depend(依賴） on the properties of the objective function(目標函數） and the variant of gradient descent used (for example, if a line search step is used). The assumptions(假設） made affect(影響） the convergence rate(速度）, and other properties, that can be proven(已證實） for gradient descent. For example, if the objective is assumed(假設的） to be strongly(非常地） convex(鼓起；中凸的） and lipschitz smooth, then gradient descent converges linearly(線性） with a fixed step size. Looser(寬鬆的） assumptions(假設） lead to either weaker convergence guarantees(保證） or require(要求） a more sophisticated(複雜的） step size selection(選擇）.

---
## References
1. https://en.wikipedia.org/wiki/Gradient_descent
2. https://books.google.com.tw/books?id=iD5s0iKXHP8C&pg=PA131&redir_esc=y#v=onepage&q&f=false
3. https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=471
---

# [返回](./LSTM.md)