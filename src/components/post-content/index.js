import React from 'react';
import './style.scss';
import { defineCustomElements as deckDeckGoHighlightElement } from "@deckdeckgo/highlight-code/dist/loader";
deckDeckGoHighlightElement();

function PostContent({ html }) {
  return (
    <div className="post-content">
      <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

export default PostContent;
