# RBM



## basic assumption

$$
\begin{flalign*}

&
\mathcal{v} \in \{0, 1\}^{d_{v} \times d_{k}}, d_{v} \ \text{is the number of rating taregt,} d_{k} \ \text{ is the scale of rating}
&

\end{flalign*}
$$

$$
\begin{flalign*}

&

\mathcal{w} \in \mathcal{R}^{d_{h} \times d_{v} \times d_{k}}, d_{h}\ \text{is the number of hidden state}

&
\end{flalign*}
$$

$$
\begin{flalign*}
&
\mathcal{h} \in \{0, 1\}^{d_{h}}
&
\end{flalign*}
$$

--------


$$
\begin{flalign*}

&
\text{For binary} \ \mathcal{v} \ \mathcal{h} \text{, define energy function:} \\

&
E(\mathcal{v}, \mathcal{h};\mathcal{w})

=  -\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 

= \mathcal{h}^{T}\mathcal{w}\mathcal{v} \\

&
\text{where}\ \mathcal{w}\mathcal{v} \ \text{is} \ einsum(\mathcal{w}, \mathcal{v}, ijk, jk \rightarrow i)
&

\end{flalign*}
$$

$$
\mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w}) = 
\frac{e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}}{Z}
\
\text{where Z is the marginal}\ \sum_{\mathcal{v}, \mathcal{h}}e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
$$


$$
\begin{flalign*}
& 
\text{Note that}\ \mathcal{h_{i}} \ \text{are supposed to be independent}.
&
\\
&
\Rightarrow \mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w}) = \prod_{i}\mathbf{P}(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}) 
&
\\
&
\text{By marginalization}, \mathbf{P}(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}) = 
\frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}}{Z_{i}} \ 

\text{where} \

Z_{i}
=\sum_{\mathcal{v}, \mathcal{h_{i}}}e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})},\
E(\mathcal{v}, \mathcal{h}_{i}; \mathcal{w})
=  -\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk} = \mathcal{h}_{i}\mathcal{w_{i}}\mathcal{v}
&

\\
\\
&
\text{cliam:} \prod_{i}Z_{i} = Z

\\

&
\text{Proof:}
E(\mathcal{v}, \mathcal{h};\mathcal{w})

=  -\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 
= \sum_{i}\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk}
= \sum_{i}E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}) \\
&
Z_{i} = \sum_{\mathcal{h}_{i}, \mathcal{v}}e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})} 
= \sum_{\mathcal{h}_{i}} \sum_{\mathcal{v}}e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})} 
= \sum_{\mathcal{h}_{i}} \sum_{\mathcal{v}}e^{\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk}} 
= \sum_{\mathcal{h}_{i}} \sum_{\mathcal{v}}e^{\mathcal{h}_{i}\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}
= \sum_{\mathcal{v}}(e^{\sum_{j, k} \mathcal{w}_{ijk}v_{jk}} + 1) \\
\\
&
\sum_{i, j, k} \ \text{means inner product, summation over each cell;} \sum_{\mathcal{v} | \mathcal{h}} \ \text{means expectation, summation over domain}
\\
\\
&
Z = \sum_{\mathcal{v}, \mathcal{h}}e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
= \sum_{\mathcal{h}}\sum_{\mathcal{v}}e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
= \sum_{\mathcal{h}}\sum_{\mathcal{v}}e^{\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk}} \\
&
= \sum_{\mathcal{h}}\sum_{\mathcal{v}}e^{\sum_{i} h_{i}\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}
= \sum_{\mathcal{h}}\sum_{\mathcal{v}}\prod_{i}e^{h_{i}\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}
= \sum_{\mathcal{h}_{1}}\cdots\sum_{\mathcal{h}_{d_{h}}}\sum_{\mathcal{v}}\prod_{i = 1}e^{h_{i}\sum_{j, k} \mathcal{w}_{ijk}v_{jk}} \\
&
= \sum_{\mathcal{h}_{2}}\cdots\sum_{\mathcal{h}_{d_{h}}}\sum_{\mathcal{v}}(e^{\sum_{j, k} \mathcal{w}_{1 jk}v_{jk}} + 1)\prod_{i = 2}e^{h_{i}\sum_{j, k} \mathcal{w}_{ijk}v_{jk}} \\
&
= \cdots \\
&
=\sum_{\mathcal{v}}\prod_{i}(e^{\sum_{j, k} \mathcal{w}_{1 jk}v_{jk}} + 1) \\
&
=\prod_{i}\sum_{\mathcal{v}}(e^{\sum_{j, k} \mathcal{w}_{1 jk}v_{jk}} + 1) \ \text{since} \ \mathcal{v} \ \text{has no dimension} \ i

\\
\\
&
\text{claim:}\ \prod_{i}e^{-E(\mathcal{v}, \mathcal{h}_{i};\mathcal{w})} = e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}
\\
&
\text{Proof:}\ \prod_{i}e^{-E(\mathcal{v}, \mathcal{h}_{i};\mathcal{w})} = \prod_{i}e^{\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk}} = e^{\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk}} = e^{-E(\mathcal{v}, \mathcal{h}; \mathcal{w})}

\end{flalign*}
$$

Further more,
$$
\mathbf{P}(\mathcal{h}|\mathcal{v};\mathcal{w}) = \prod_{i}\mathbf{P}(\mathcal{h}_{i}|\mathcal{v};\mathcal{w}) \\

\mathbf{P}(\mathcal{v}|\mathcal{h}; \mathcal{w}) = \prod_{j}\mathbf{P}(\mathcal{v}_{j\cdot} | \mathcal{h}; \mathcal{w})
$$
### Propagation

For the condition probability,
$$
\begin{align*}

\mathbf{P}(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w}) 
&= \frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}}{Z_{i}}
\\

Z_{i}
&=\sum_{\mathcal{v}, \mathcal{h_{i}}}e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})},
\\

E(\mathcal{v}, \mathcal{h}_{i}; \mathcal{w})

&=  -\sum_{j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 
\\
&= -\mathcal{h}_{i}\mathcal{w_{i}}\mathcal{v}
\\

\\

\mathbf{P}(\mathcal{h}_{i} = 1|\mathcal{v};\mathcal{w}) 

&= \frac{\mathbf{\mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w})}}{\mathbf{P}(\mathcal{h}_{i} = 0, \mathcal{v};\mathcal{w}) + \mathbf{P}(\mathcal{h}_{i} = 1, \mathcal{v};\mathcal{w})}
\\

&= \frac{\frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}}{Z_{i}}|_{\mathcal{h}_{i} = 1}}{\frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}}{Z_{i}}|_{\mathcal{h}_{i} = 0} + \frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}}{Z_{i}}|_{\mathcal{h}_{i} = 1}}
\\

&= \frac{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 1}}{e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 0} + e^{-E(\mathcal{v}, \mathcal{h_{i}}; \mathcal{w})}|_{\mathcal{h}_{i} = 1}}
\\

&= \frac{e^{\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}}{e^{0} + e^{\sum_{j, k} \mathcal{w}_{ijk}v_{jk}}}
\\

&=\frac{1}{e^{-\sum_{j, k} \mathcal{w}_{ijk}v_{jk}} + 1} \\

&= \sigma(\sum_{j, k} \mathcal{w}_{ijk}v_{jk}) \ \text{where} \ \sigma(x) = \frac{1}{1 + e^{-x}}
\\

& = \sigma(\sum_{j} \mathcal{w}_{ij\cdot} \cdot v_{j\cdot}),\ \text{inner production on dimension k}


\end{align*}
$$

$$
\begin{align*}

\mathbf{P}(\mathcal{v}_{jk} = 1|\mathcal{h}; \mathcal{w}) 

& = \frac
{\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})}
{\sum_{k^{\prime} \neq k}\mathbf{P}(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j,-k^{\prime}} = 0, \mathcal{h}; \mathcal{w}) +\mathbf{P}(\mathcal{v}_{jk} = 1, \mathcal{h}; \mathcal{w})} \ \text{since each  row of $\mathcal{v}$ is NOT independent, it's one-hot} \\

& =  \frac
{
	\frac
		{e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}}
		{Z_{j}}
}
{
	\sum_{k_{\prime} \neq k}\frac
		{
			e^{-E(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j, -k^{\prime}} = 0, \mathcal{h}; 					\mathcal{w})}
		}
		{Z_{j}}
	+
	\frac
		{e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}}
		{Z_{j}}
}
\
\text{where $Z_{j}$ is $\sum_{\mathcal{v}_{j\cdot}}e^{-E(\mathcal{v}_{j\cdot},\mathcal{h}; \mathcal{w})}$}

\\

& = \frac
{
	e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}
}
{
	\sum_{k_{\prime} \neq k} e^{-E(\mathcal{v}_{jk^{\prime}} = 1, \mathcal{v}_{j, -k^{\prime}} = 0, 		\mathcal{h}; \mathcal{w})}
	+
	e^{-E(\mathcal{v}_{jk} = 1, \mathcal{v}_{j, -k} = 0, \mathcal{h}; \mathcal{w})}
}
\\

& = \frac
{
	e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk}}
}
{
	\sum_{k^{\prime} \neq k}e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk^{\prime}}} + 							e^{\mathcal{h}^{T} \cdot \mathcal{w}_{\cdot jk}}
}
\\

& = Softmax(\sum_{i}\mathcal{h}_{i}\mathcal{w}_{ijk})
\
\text{as project description note}


\end{align*}
$$

### Learning the weight

$$
\text{The obejctive is to minimize the KL divergence between $\mathbf{P}_{data}(\mathcal{v})$ and $\mathbf{P}(\mathcal{v}; \mathcal{w})$.}
\\
\text{The later is $\sum_{\mathcal{h}}\mathbf{P}(\mathcal{v}, \mathcal{h}; \mathcal{w})$, the marginal distribution of $\mathcal{v}$ from the joint distribution parameterized by $\mathcal{w}$}
$$

cited from "An Overview of Restricted Boltzmann Machines"
$$
\text{suppose we have N training samples, $\tau = \{\mathcal{v}^{(1)}, \cdots, \mathcal{v}^{(N)}\}$}
\\

\begin{align*}
d_{kl}(\mathbf{P}_{data}(v) || \mathbf{P}(\mathcal{v}; \mathcal{w}))

&= \sum_{v \in \tau} \mathbf{P}_{data}(v) \log(\frac{\mathbf{P}_{data}(v)}{\mathbf{P}(\mathcal{v}; \mathcal{w})})
\\

&= \sum_{v \in \tau} 
[ 
	\mathbf{P}_{data}(v) \log(\mathbf{P}_{data}(v)) - \mathbf{P}_{data}(v) \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
]
\end{align*}
$$

$$
\text{minimizing the KL divergence is equavilent to}
\\

\begin{align*}

\arg \min_{\mathcal{w}} d_{kl}(\mathbf{P}_{data}(v) || \mathbf{P}(\mathcal{v}; \mathcal{w}))

& = \arg \min_{\mathcal{w}}
\sum_{v \in \tau} 
[ 
	\mathbf{P}_{data}(v) \log(\mathbf{P}_{data}(v)) - \mathbf{P}_{data}(v) \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
]
\\

&= \arg \max_{\mathcal{w}} 
\sum_{v \in \tau} \mathbf{P}_{data}(\mathcal{v}) \log(\mathbf{P}(\mathcal{v}; \mathcal{w})
\\

&= \sum_{v \in \tau} \mathbf{P}_{data}(\mathcal{v})
\arg \max_{\mathcal{w}} \log(\mathbf{P}(\mathcal{v}; \mathcal{w}))
\
\text{when training}


\end{align*}
\\

\text{where $\mathbf{P}_{data}(\mathcal{v}) = \frac{1}{N}\sum_{i=1}^{N} \delta(\mathcal{v} - \mathcal{v}_{i})$, $\delta$ is the dirac delta function}
$$

$$
\begin{align*}

\ln \mathbf{P}(v; w) 

&= \ln \sum_{h} \mathbf{P}(v, h;w)
\\

&= \ln \sum_{h} \frac{1}{Z} e^{-E(v, h; w)} = \ln \frac{\sum_{h} e^{-E(v, h; w)}}{Z}
\\

&= \ln \sum_{h}e^{-E(v, h; w)} - \ln \sum_{v, h}e^{-E(v, h; w)}


\end{align*}
$$

Take gradient of each part w.r.t $w $

The former part
$$
\begin{align*}

\frac{\partial}{\partial w_{ijk}} \ln \sum_{h}e^{-E(v, h; w)} &=  \frac{1}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{h}e^{-E(v, h; w)}

\\
&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{h}e^{-E(v, h; w)}
\\

&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \sum_{h} \frac{\partial}{\partial w_{ijk}} e^{-E(v, h; w)}
\\

&= \frac{1}{\sum_{h}e^{-E(v, h; w)}} \sum_{h} e^{-E(v, h;w)} \frac{\partial}{\partial w_{ijk}} (-E(v, h; w))
\\

&=- \frac{\sum_{h} e^{-E(v, h;w)} \frac{\partial}{\partial w_{ijk}} E(v, h; w)}{\sum_{h}e^{-E(v, h; w)}}
\\

&=- \sum_{h} \frac{e^{-E(v, h;w)}}{\sum_{h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&=- \sum_{h} \frac{\frac{e^{-E(v, h;w)}}{Z}}{\sum_{h}  \frac{e^{-E(v, h; w)}}{Z} } \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&= - \sum_{h}\mathbf{P}(h|v;w) \frac{\partial}{\partial w_{ijk}} E(v, h; w)
\\

&= \sum_{h}\mathbf{P}(h|v;w) \frac{\partial}{\partial w_{ijk}} \sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk} 
\\

&= \sum_{h}\mathbf{P}(h|v;w) h_{i}v_{jk} 
\\

&= \sum_{h_{i} \in \{0, 1\}} \mathbf{P}(h_{i}|v;w) h_{i}v_{jk} \sum_{h_{-i}}\mathbf{P}(h_{i}|v;w) 
\\

&= \mathbf{P}(h_{i} = 1|v;w) v_{jk} \sum_{h_{-i}}\mathbf{P}(h_{i}|v;w)
\\

&= \mathbf{P}(h_{i} = 1|v;w) v_{jk} \
\text{
since 
$
\sum_{h_{-i}}\mathbf{P}(h_{i}|v;w) = \sum_{h_{i^{\prime}}}
\frac{ \prod_{i^{\prime} \neq i} \mathbf{P}(h_{i^{\prime}}|v;w)}{\sum_{h_{i^{\prime}}} \prod_{i^{\prime} \neq i} \mathbf{P}(h_{i^{\prime}}|v;w)}
= 1
$
}


\end{align*}
$$

The latter part
$$
\begin{align*}

\frac{\partial}{\partial w_{ijk}}  - \ln \sum_{v, h}e^{-E(v, h; w)} 
&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \frac{\partial}{\partial w_{ijk}} \sum_{v, h}e^{-E(v, h; w)}
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)} \frac{\partial}{\partial w_{ijk}} (-E(v, h; w))
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)} \frac{\partial}{\partial w_{ijk}}
\sum_{i, j, k} \mathcal{w}_{ijk}h_{i}v_{jk}
\\

&= - \frac{1}{\sum_{v, h}e^{-E(v, h; w)}} \sum_{v, h} e^{-E(v, h; w)}h_{i}v_{jk}
\\

&= - \sum_{v, h} \frac{e^{-E(v, h; w)}}{Z} h_{i}v_{jk}
\\

&= - \sum_{v, h} \mathbf{P}(v, h; w) h_{i}v_{jk}
\\

&= - \sum_{v} \mathbf{P}(v; w) \textcolor{blue} {\sum_{h} \mathbf{P}(h|v;w)h_{i}v_{jk}}
\
\text{where the blue part is the same as the former}
\\

&= -\mathbf{E}_{v \sim \mathbf{P}(v;w)} [ \mathbf{P}(h_{i} = 1|v;w) v_{jk} ]
\\

&= -\mathbf{P}(h_{i} = 1|\bar{v};w) \bar{v}_{jk} \
\text{where $\bar{v}$ is the predicted $v$}

\end{align*}
$$
