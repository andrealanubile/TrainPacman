import { Text, TextStyle } from 'pixi.js';

class ScoreText {
    constructor(scaleX = 1, scaleY = 1) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
        this.score = 0
        this.fontSize = 18;
        const score_str = String(this.score).padStart(8, '0');
        this.style = new TextStyle({
            fontFamily: 'monospace',
            fontSize: this.fontSize * scaleY,
            // fontStyle: 'italic',
            fontWeight: 'bold',
            fill: 'white',
        });
        this.text = new Text({
            text: `SCORE\n${score_str}`,
            style: this.style
        });
        this.text.x = 5
        this.text.y = 5
    }

    updateScore(score) {
        this.score = score;
        const score_str = String(this.score).padStart(8, '0');
        this.text.text = `SCORE\n${score_str}`
        this.text.style.fontSize = this.fontSize * this.scaleY;
    }

    getText() {
        return this.text;
    }

    updateScale(scaleX, scaleY) {
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }
}

export default ScoreText;