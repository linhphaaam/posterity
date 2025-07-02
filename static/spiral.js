const SPIRAL_SCALE = 4; // Scale factor for spiral coordinates
const MAX_IMAGE_SIZE = 80; // Maximum width/height for images
let centerImg;
let posterImages = [];
let decadeData = []; // Decade distribution data
let highlightedDecade = null; // Highlighted decade
let selectedPoster = null;
let spiral_x_offset = 200;
let chartWidth_scale = 1.6

function preload() {
    centerImg = loadImage(centerImage);
    spiralData.forEach(poster => {
        posterImages.push({
            img: loadImage(poster.image_path),
            x: poster.x * SPIRAL_SCALE,
            y: poster.y * SPIRAL_SCALE,
            title: poster.title,
            similarity: poster.similarity,
            decade: poster.decade
        });
    });

    // Load decade data from Flask-rendered JSON
    decadeData = JSON.parse(decadeDataJson);
}

function setup() {
    const canvas = createCanvas(1400, 1200);
    canvas.parent('spiralCanvas');
    imageMode(CENTER);
    textAlign(CENTER, CENTER);
}

function draw() {
    background(255);

    const centerX = width / 2 - spiral_x_offset;
    const centerY = height / 3;

    drawCenterImage(centerX, centerY);
    // drawLinesToPosters(centerX, centerY);
    drawSpiralSpine(centerX, centerY); // Unchanged spine
    drawPosters(centerX, centerY);
    drawLinearDecadeChart();

    if (selectedPoster) {
        drawPosterModal();
    }
}

function drawCenterImage(centerX, centerY) {
    const aspectRatio = centerImg.width / centerImg.height;
    let centerImgW, centerImgH;

    if (centerImg.width > centerImg.height) {
        centerImgW = 120;
        centerImgH = centerImgW / aspectRatio;
    } else {
        centerImgH = 160;
        centerImgW = centerImgH * aspectRatio;
    }

    image(centerImg, centerX, centerY, centerImgW, centerImgH);
}

// function drawLinesToPosters(centerX, centerY) {
//     for (let poster of posterImages) {
//         const imgX = centerX + poster.x;
//         const imgY = centerY + poster.y;

//         // Line from center to posters (always visible)
//         strokeWeight(1);
//         stroke(200, 0, 0, 100);
//         line(centerX, centerY, imgX, imgY);

//         // Display similarity value along the line
//         noStroke();
//         fill(0);
//         textSize(12);
//         text(poster.similarity, (centerX + imgX) / 2, (centerY + imgY) / 2);
//     }
// }

function drawSpiralSpine(centerX, centerY) {
    noFill();
    stroke(0, 70);
    strokeWeight(1);

    beginShape();
    for (let poster of posterImages) {
        const imgX = centerX + poster.x;
        const imgY = centerY + poster.y;
        curveVertex(imgX, imgY); // Always visible
    }
    endShape();
}

function drawPosters(centerX, centerY) {
    for (let poster of posterImages) {
        const imgX = centerX + poster.x;
        const imgY = centerY + poster.y;

        const transparency = highlightedDecade && poster.decade !== highlightedDecade ? 50 : 255;

        // Maintain aspect ratio
        const aspectRatio = poster.img.width / poster.img.height;
        let imgW = MAX_IMAGE_SIZE, imgH = MAX_IMAGE_SIZE;
        if (poster.img.width > poster.img.height) {
            imgH = MAX_IMAGE_SIZE / aspectRatio;
        } else {
            imgW = MAX_IMAGE_SIZE * aspectRatio;
        }

        tint(255, transparency); // Adjust transparency
        image(poster.img, imgX, imgY, imgW, imgH);
        noTint(); // Reset tint
    }
}


function drawLinearDecadeChart() {
    const chartX = 100; // Chart starting X position
    const chartY = height - 200; // Chart starting Y position
    const chartWidth = width / chartWidth_scale; // Total width for the chart
    const chartHeight = 150; // Total height for the chart

    noFill();
    noStroke();
    rect(chartX, chartY, chartWidth, chartHeight);

    // Add chart title
    fill(0);
    textSize(20);
    textAlign(CENTER, CENTER);
    text("Decade Distribution of Posters", chartX + chartWidth / 2, chartY - 30); // Title above the chart

    const decades = decadeData.map(d => d[0]);
    const counts = decadeData.map(d => d[1]);
    const minDecade = Math.min(...decades);
    const maxDecade = Math.max(...decades);

    const decadeRange = maxDecade - minDecade + 1;
    const barWidth = (chartWidth / decadeRange) * 10; // Increased bar width
    const maxCount = Math.max(...counts);

    for (let i = 0; i < decades.length; i++) {
        const decade = decades[i];
        const count = counts[i];

        const xPos = map(decade, minDecade, maxDecade, chartX, chartX + chartWidth);
        const barHeight = map(count, 0, maxCount, 0, chartHeight - 20);

        // Highlight selected decade
        if (highlightedDecade === decade) {
            fill('#FF7043'); // Highlight color
        } else {
            fill('#000000'); // Default bar color
        }

        // Draw bar
        rect(xPos - barWidth / 2, chartY + chartHeight - barHeight, barWidth, barHeight);

        // Display count above bar
        fill(0);
        noStroke();
        textSize(18);
        text(count, xPos, chartY + chartHeight - barHeight - 10);

        // Display decade below bar
        text(decade, xPos, chartY + chartHeight + 15);
    }

    // Draw x-axis
    stroke(0);
    strokeWeight(1);
    line(chartX - (barWidth / 2), chartY + chartHeight, chartX + chartWidth + (barWidth / 2), chartY + chartHeight);
}

const posterModal_y_offset = 700;

function drawPosterModal() {
    fill(255);
    stroke(0);
    rect(width - 330, height - posterModal_y_offset, 320, 420, 10);

    fill(0);
    textSize(16);
    textAlign(LEFT, TOP);
    text(`Title: ${selectedPoster.title}`, width - 320, height - (posterModal_y_offset - 320));
    text(`Similarity: ${selectedPoster.similarity}`, width - 320, height - (posterModal_y_offset - 350));
    text(`Decade: ${selectedPoster.decade}`, width - 320, height - (posterModal_y_offset - 380));

    const aspectRatio = selectedPoster.img.width / selectedPoster.img.height;
    let imgW = 300, imgH = 300;
    if (selectedPoster.img.width > selectedPoster.img.height) {
        imgH = 300 / aspectRatio;
    } else {
        imgW = 300 * aspectRatio;
    }

    image(selectedPoster.img, width - 170, height - (posterModal_y_offset - 160), imgW, imgH);
}

function mousePressed() {
    const centerX = width / 2 - spiral_x_offset;
    const centerY = height / 3;

    // Check poster selection
    for (let poster of posterImages) {
        const imgX = centerX + poster.x;
        const imgY = centerY + poster.y;

        if (dist(mouseX, mouseY, imgX, imgY) < 40) {
            selectedPoster = poster;
            return;
        }
    }
    selectedPoster = null;

    // Check decade bar selection
    const chartY = height - 200;
    const chartHeight = 150;
    const chartX = 50;
    const chartWidth = width / chartWidth_scale;

    const minDecade = Math.min(...decadeData.map(d => d[0]));
    const maxDecade = Math.max(...decadeData.map(d => d[0]));

    for (let i = 0; i < decadeData.length; i++) {
        const decade = decadeData[i][0];
        const xPos = map(decade, minDecade, maxDecade, chartX, chartX + chartWidth);
        const barWidth = (chartWidth / (maxDecade - minDecade + 1)) * 10;

        if (mouseX > xPos - barWidth / 2 && mouseX < xPos + barWidth / 2 &&
            mouseY > chartY && mouseY < chartY + chartHeight) {
            highlightedDecade = highlightedDecade === decade ? null : decade; // Toggle
            return;
        }
    }

    highlightedDecade = null; // Reset if clicking outside bars
}
