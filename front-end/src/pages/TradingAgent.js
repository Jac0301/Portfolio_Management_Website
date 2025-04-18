
import React from 'react';
import TradingPerformance from './TradingPerformance';

const API_URL = process.env.REACT_APP_API_URL;
const TradingAgent = () => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showStockName, setShowStockName] = useState(false);
  const [dateRange, setDateRange] = useState([0, 100]); // 百分比值
  const [allDates, setAllDates] = useState([]); // 存儲所有可用日期
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_URL}/trading-performance`);
        setPerformanceData(response.data);
        // 設置所有可用日期
        const dates = response.data.account_value.map(item => item.date);
        setAllDates(dates);
        setDateRange([0, 100]); // 初始顯示全部範圍
        setLoading(false);
      } catch (err) {
        setError('獲取數據失敗');
        setLoading(false);
        console.error('獲取數據錯誤:', err);
      }
    };
    
    fetchData();
  }, []);
  
  // 獲取股票顯示名稱
  const getStockDisplayName = (stockCode) => {
    if (showStockName) {
      // 從股票代號中移除 '.TW' 後綴
      const code = stockCode.replace('.TW', '');
      return stockNames[`${code}.TW`] || code;
    }
    return stockCode;
  };

  // 獲取股票的產業顏色
  const getStockColor = (stockCode) => {
    const industry = stockIndustries[stockCode + '.TW'];
    return industry ? industryColors[industry] : '#666';
  };

  // 產業分類標籤
  const renderIndustryLabels = () => {
    if (!performanceData) return null;
    
    // 獲取所有出現的股票的產業
    const industries = new Set();
    performanceData.stocks.forEach(stock => {
      const stockCode = stock + '.TW';
      const industry = stockIndustries[stockCode];
      if (industry) industries.add(industry);
    });

    return Array.from(industries).map(industry => (
      <span 
        key={industry} 
        className={`industry-label industry-${industry.replace(/\s+/g, '-').toLowerCase()}`}
      >
        {industry} ({
          performanceData.stocks.filter(stock => 
            stockIndustries[stock + '.TW'] === industry
          ).length
        })
      </span>
    ));
  };

  // 準備圖表數據
  const prepareChartData = () => {
    if (!performanceData) return [];
    
    // 合併帳戶價值和交易行為數據
    return performanceData.account_value.map(dayValue => {
      const dayActions = performanceData.actions.find(a => a.date === dayValue.date) || {};
      
      // 創建包含所有數據的對象
      const dayData = {
        date: dayValue.date,
        cumulative_return: dayValue.cumulative_return
      };
      
      // 公司視圖：按交易量排序股票
      const buyStocks = [];
      const sellStocks = [];
      
      // 分類買入和賣出股票
      performanceData.stocks.forEach(stock => {
        const action = dayActions[stock] || 0;
        if (action > 0) {
          buyStocks.push({ stock, action });
        } else if (action < 0) {
          sellStocks.push({ stock, action });
        }
      });
      
      // 按交易量絕對值排序（大的在前）
      buyStocks.sort((a, b) => Math.abs(b.action) - Math.abs(a.action));
      sellStocks.sort((a, b) => Math.abs(b.action) - Math.abs(a.action));
      
      // 添加排序後的買入股票
      buyStocks.forEach((item, index) => {
        dayData[`buy_${index}_${item.stock}`] = item.action;
      });
      
      // 添加排序後的賣出股票
      sellStocks.forEach((item, index) => {
        dayData[`sell_${index}_${item.stock}`] = item.action;
      });
      
      return dayData;
    });
  };

  // 獲取圖表的Bar組件
  const renderBars = () => {
    if (!chartData || chartData.length === 0) return null;
    
    // 找出所有買入和賣出的鍵
    const buyKeys = new Set();
    const sellKeys = new Set();
    
    chartData.forEach(day => {
      Object.keys(day).forEach(key => {
        if (key.startsWith('buy_')) buyKeys.add(key);
        if (key.startsWith('sell_')) sellKeys.add(key);
      });
    });
    
    // 渲染買入和賣出的Bar
    return [
      // 買入Bar
      ...Array.from(buyKeys).map(key => {
        const stockCode = key.split('_')[2];
        return (
          <Bar
            key={key}
            yAxisId="left"
            dataKey={key}
            name={getStockDisplayName(stockCode)}
            fill={getStockColor(stockCode)}
            opacity={0.6}
            stackId="positive"
          />
        );
      }),
      // 賣出Bar
      ...Array.from(sellKeys).map(key => {
        const stockCode = key.split('_')[2];
        return (
          <Bar
            key={key}
            yAxisId="left"
            dataKey={key}
            name={getStockDisplayName(stockCode)}
            fill={getStockColor(stockCode)}
            opacity={0.6}
            stackId="negative"
          />
        );
      })
    ];
  };

  // 自定義Tooltip內容
  const renderTooltip = ({ payload, label, active }) => {
    if (!active || !payload || payload.length === 0) return null;
    
    // 過濾掉交易量為0的項目
    const filteredPayload = payload.filter(entry => 
      entry.name !== "累積報酬率" && entry.value !== 0 && entry.value !== null
    );
    
    // 獲取累積報酬率
    const returnEntry = payload.find(p => p.name === "累積報酬率");
    
    // 按公司分組
    const groupedItems = {};
    
    filteredPayload.forEach(entry => {
      let name;
      let value = entry.value;
      
      // 從key中提取股票代碼
      const parts = entry.dataKey.split('_');
      name = getStockDisplayName(parts[2]);
      
      // 合併同名項目
      if (groupedItems[name]) {
        groupedItems[name].value += value;
        groupedItems[name].color = entry.color;
      } else {
        groupedItems[name] = { value, color: entry.color };
      }
    });
    
    return (
      <div className="custom-tooltip">
        <p>{`日期: ${label}`}</p>
        {returnEntry && (
          <p>{`累積報酬率: ${(returnEntry.value * 100).toFixed(2)}%`}</p>
        )}
        {Object.entries(groupedItems)
          .sort(([, a], [, b]) => Math.abs(b.value) - Math.abs(a.value))
          .map(([name, { value, color }], index) => (
            <p key={index} className="tooltip-item" data-color={color}>
              {name}: {value}
            </p>
          ))
        }
      </div>
    );
  };

  // 處理日期範圍變化
  const handleDateRangeChange = (range) => {
    setDateRange(range);
  };

  // 根據日期範圍過濾數據
  const getFilteredChartData = () => {
    if (!chartData || chartData.length === 0) return [];
    
    const startIndex = Math.floor(chartData.length * dateRange[0] / 100);
    const endIndex = Math.ceil(chartData.length * dateRange[1] / 100);
    
    return chartData.slice(startIndex, endIndex);
  };

  // 格式化日期顯示
  const formatDate = (date) => {
    return date.replace(/(\d{4})-(\d{2})-(\d{2})/, '$1/$2/$2');
  };

  // 修改表格渲染函數，新增累積報酬率欄位
  const renderDailyTable = () => {
    if (!performanceData) return null;

    const filteredData = getFilteredChartData();
    
    return (
      <div className="daily-table">
        <table>
          <thead>
            <tr>
              <th>日期</th>
              <th>資產總值</th>
              <th>當日報酬率</th>
              <th>累積報酬率</th>
              <th>交易行為</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map(day => {
              // 找出當天的交易行為
              const trades = [];
              Object.entries(day).forEach(([key, value]) => {
                if ((key.startsWith('buy_') || key.startsWith('sell_')) && value !== 0) {
                  const [action, , stock] = key.split('_');
                  trades.push({
                    action: action === 'buy' ? '買入' : '賣出',
                    stock: showStockName ? getStockDisplayName(stock) : stock,
                    value: Math.abs(value)
                  });
                }
              });

              // 找出對應的資產價值
              const accountValue = performanceData.account_value.find(av => av.date === day.date);

              return (
                <tr key={day.date}>
                  <td>{day.date}</td>
                  <td>{accountValue ? accountValue.account_value.toLocaleString() : '-'}</td>
                  <td className={accountValue?.daily_return > 0 ? 'positive' : 'negative'}>
                    {accountValue ? `${(accountValue.daily_return * 100).toFixed(2)}%` : '-'}
                  </td>
                  <td className={day.cumulative_return > 0 ? 'positive' : 'negative'}>
                    {`${(day.cumulative_return * 100).toFixed(2)}%`}
                  </td>
                  <td>
                    {trades.length > 0 ? (
                      <div className="trades-list">
                        {trades.map((trade, index) => (
                          <span 
                            key={index} 
                            className={trade.action === '買入' ? 'buy' : 'sell'}
                          >
                            {`${trade.action} ${trade.stock}(${trade.value})`}
                          </span>
                        ))}
                      </div>
                    ) : '-'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  if (loading) return <div className="trading-performance-loading">數據加載中...</div>;
  if (error) return <div className="trading-performance-error">{error}</div>;
  if (!performanceData) return <div className="trading-performance-error">無法獲取數據</div>;
  
  const chartData = prepareChartData();
  
  return (
    <div style={{ 
      backgroundColor: 'white',
      borderRadius: '8px',
      padding: '20px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      marginBottom: '20px'
    }}>
      <TradingPerformance />
    </div>
  );
};

export default TradingAgent; 