document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-file');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const alphaValueSpan = document.getElementById('alpha-value');
    const analysisText = document.getElementById('analysis-text');
    let spectrumChart = null;

    fileInput.addEventListener('change', () => {
        analyzeBtn.disabled = !fileInput.files.length;
    });

    analyzeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        // UI制御
        analyzeBtn.disabled = true;
        loadingDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');

        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // 短い音声の警告
            if (audioBuffer.duration < 5.0) {
                alert("⚠️ 音声が短すぎます。安定した1/fゆらぎ解析のため、5秒以上の音声を推奨します。");
            }

            processAudio(audioBuffer);
            
        } catch (error) {
            console.error(error);
            alert("音声ファイルの解析に失敗しました。対応している形式か確認してください。");
        } finally {
            analyzeBtn.disabled = false;
            loadingDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');
        }
    });

    function processAudio(audioBuffer) {
        // 1. モノラル化 (Lチャンネルを使用)
        const channelData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;

        // 波形描画
        drawWaveform(channelData);

        // 2. 特徴量抽出 (RMS時系列の作成)
        const frameSize = 2048; // 約46ms
        const hopSize = 1024;   // 約23ms
        const rmsData = [];

        for (let i = 0; i < channelData.length - frameSize; i += hopSize) {
            let sum = 0;
            for (let j = 0; j < frameSize; j++) {
                sum += channelData[i + j] * channelData[i + j];
            }
            rmsData.push(Math.sqrt(sum / frameSize));
        }

        // 3. トレンド除去と窓関数 (ノイズ耐性)
        const meanRms = rmsData.reduce((a, b) => a + b) / rmsData.length;
        let signal = rmsData.map(val => val - meanRms);
        applyHanningWindow(signal);

        // FFTのために2の累乗にパディング/切り捨て
        const N = Math.pow(2, Math.floor(Math.log2(signal.length)));
        signal = signal.slice(0, N);

        // 4. スペクトル解析 (簡易FFT)
        const { real, imag } = fft(signal);
        
        // パワースペクトル算出 (ナイキスト周波数まで)
        const halfN = N / 2;
        const frameRate = sampleRate / hopSize;
        const freqStep = frameRate / N;
        
        const logF = [];
        const logP = [];
        const chartData = [];

        // 5. 1/f解析 (ノイズ対策: 極端な低周波と高周波を除外)
        // 例: 0.2Hz 〜 frameRate/4 (おおよそ10Hz) までを解析対象とする
        for (let i = 1; i < halfN; i++) {
            const f = i * freqStep;
            const power = (real[i] * real[i] + imag[i] * imag[i]) / N;

            if (f > 0.2 && f < frameRate / 4 && power > 0) {
                const x = Math.log10(f);
                const y = Math.log10(power);
                logF.push(x);
                logP.push(y);
                chartData.push({ x: x, y: y });
            }
        }

        // 最小二乗法による直線フィット
        const { slope, intercept } = linearRegression(logF, logP);
        const alpha = -slope; // P(f) ∝ 1/f^α -> log P = -α log f + C

        // 6. 表示処理
        displayResults(alpha);
        drawChart(chartData, slope, intercept);
    }

    function displayResults(alpha) {
        alphaValueSpan.textContent = alpha.toFixed(3);
        
        // 連続値に基づく解説
        let text = "";
        let color = "";
        if (alpha >= 0.8 && alpha <= 1.2) {
            text = "🌟 1/fゆらぎに近く、心地よく自然な揺らぎを持っています。";
            color = "#2e7d32"; // Green
        } else if (alpha < 0.5) {
            text = "🎲 ランダム性が強い傾向があります（ホワイトノイズに近い）。";
            color = "#1565c0"; // Blue
        } else if (alpha >= 1.5) {
            text = "〰️ 変化が少なく、滑らかで単調な傾向があります（ブラウンノイズに近い）。";
            color = "#e65100"; // Orange
        } else if (alpha >= 0.5 && alpha < 0.8) {
            text = "ややランダム性が強いですが、自然な揺らぎの片鱗があります。";
            color = "#2c3e50";
        } else {
            text = "単調な傾向が強めですが、適度な揺らぎも含まれています。";
            color = "#2c3e50";
        }

        analysisText.textContent = text;
        analysisText.style.color = color;
    }

    // --- ユーティリティ関数 ---

    // Hanning窓
    function applyHanningWindow(data) {
        const n = data.length;
        for (let i = 0; i < n; i++) {
            data[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
        }
    }

    // 最小二乗法
    function linearRegression(x, y) {
        const n = x.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        for (let i = 0; i < n; i++) {
            sumX += x[i]; sumY += y[i];
            sumXY += x[i] * y[i]; sumXX += x[i] * x[i];
        }
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        return { slope, intercept };
    }

    // 軽量Radix-2 FFT (外部依存排除用)
    function fft(inputReal) {
        const n = inputReal.length;
        const real = new Float32Array(inputReal);
        const imag = new Float32Array(n);

        // Bit-reversal
        let j = 0;
        for (let i = 0; i < n - 1; i++) {
            if (i < j) {
                let tr = real[i], ti = imag[i];
                real[i] = real[j]; imag[i] = imag[j];
                real[j] = tr; imag[j] = ti;
            }
            let m = n >> 1;
            while (j >= m) { j -= m; m >>= 1; }
            j += m;
        }

        // Cooley-Tukey
        for (let size = 2; size <= n; size *= 2) {
            const halfsize = size / 2;
            const tablestep = n / size;
            const angleStep = -2 * Math.PI / size;
            for (let i = 0; i < n; i += size) {
                for (let j = i, k = 0; j < i + halfsize; j++, k++) {
                    const angle = k * angleStep;
                    const cosA = Math.cos(angle);
                    const sinA = Math.sin(angle);
                    const tpre = real[j + halfsize] * cosA - imag[j + halfsize] * sinA;
                    const tpim = real[j + halfsize] * sinA + imag[j + halfsize] * cosA;
                    real[j + halfsize] = real[j] - tpre;
                    imag[j + halfsize] = imag[j] - tpim;
                    real[j] += tpre;
                    imag[j] += tpim;
                }
            }
        }
        return { real, imag };
    }

    // 波形描画
    function drawWaveform(channelData) {
        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        
        ctx.beginPath();
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 1;
        
        // 描画が重くならないよう間引いて描画
        const step = Math.ceil(channelData.length / width);
        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const idx = i * step + j;
                if (idx < channelData.length) {
                    const datum = channelData[idx];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }
            }
            const yMin = ((1 - min) / 2) * height;
            const yMax = ((1 - max) / 2) * height;
            ctx.moveTo(i, yMin);
            ctx.lineTo(i, yMax);
        }
        ctx.stroke();
    }

    // グラフ描画 (Chart.js)
    function drawChart(scatterData, slope, intercept) {
        const ctx = document.getElementById('spectrum-chart').getContext('2d');
        if (spectrumChart) spectrumChart.destroy();

        // 回帰直線のデータポイント作成
        const minX = Math.min(...scatterData.map(d => d.x));
        const maxX = Math.max(...scatterData.map(d => d.x));
        const lineData = [
            { x: minX, y: slope * minX + intercept },
            { x: maxX, y: slope * maxX + intercept }
        ];

        spectrumChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'パワースペクトル (RMS)',
                        data: scatterData,
                        backgroundColor: 'rgba(76, 175, 80, 0.5)',
                        pointRadius: 2
                    },
                    {
                        type: 'line',
                        label: `回帰直線 (α = ${(-slope).toFixed(2)})`,
                        data: lineData,
                        borderColor: '#e53935',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'log10( Frequency )' } },
                    y: { title: { display: true, text: 'log10( Power )' } }
                },
                animation: false
            }
        });
    }
});